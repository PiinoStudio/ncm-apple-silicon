import gc
import time
from typing import Dict, Optional, Union

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ncm_mps_runner import get_apple_silicon_device
from src.telemetry import get_peak_memory_gb


class AutoregressiveBaselineRunner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        console: Optional[Console] = None
    ):
        self.model_id = model_id
        self.device = device or get_apple_silicon_device()
        self.dtype = dtype
        self.console = console or Console()

        self._load_model()

    def _load_model(self):
        self.console.print(f"[bold cyan][AR Baseline][/bold cyan] Loading [yellow]{self.model_id}[/yellow] on [green]{self.device}[/green]...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=None
        ).to(self.device).eval()

        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.model_params_m = self.total_params / 1e6
        elem_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        self.model_size_gb = (self.total_params * elem_size) / (1024 ** 3)
        self.console.print(f"  ✓ AR Model Loaded: {self.model_params_m:.1f}M params ({self.model_size_gb:.2f} GB in RAM)")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> Dict[str, any]:
        """
        Executes standard autoregressive generation token-by-token.
        NFE is strictly linear: 1 forward pass per generated token.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        t_start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id
        )
        if self.device.type == "mps":
            torch.mps.synchronize()
        t_end = time.perf_counter()

        gen_tokens = outputs.shape[1] - prompt_len
        wall_time = max(t_end - t_start, 1e-6)
        tok_s = gen_tokens / wall_time

        # In autoregressive decoding with batch_size=1, every token generated
        # reads the full model parameters from memory over the bus.
        nfe = gen_tokens
        est_bandwidth_gb = nfe * self.model_size_gb

        decoded = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        return {
            "text": decoded,
            "prompt_tokens": prompt_len,
            "generated_tokens": gen_tokens,
            "nfe": nfe,
            "wall_time_s": wall_time,
            "tokens_per_sec": tok_s,
            "model_params_m": self.model_params_m,
            "model_size_gb": self.model_size_gb,
            "est_bandwidth_gb": est_bandwidth_gb,
            "gb_per_token": est_bandwidth_gb / max(gen_tokens, 1)
        }

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

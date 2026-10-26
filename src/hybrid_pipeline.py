import gc
import os
import time
from typing import Dict, Optional, Union

import torch
from rich.console import Console

from src.ncm_mps_runner import NCMRunner
from src.telemetry import TelemetryContext, get_peak_memory_gb


class HybridPipelineRunner:
    def __init__(
        self,
        mlx_model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        fallback_mlx_model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        console: Optional[Console] = None
    ):
        self.mlx_model_id = mlx_model_id
        self.fallback_mlx_model_id = fallback_mlx_model_id
        self.console = console or Console()

    def run_stage1_ncm(self, raw_input: Union[str, list], task: str = "xsum") -> Dict[str, any]:
        """Stage 1: NCM rapid hierarchical concept drafting on MPS."""
        self.console.print("\n[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
        self.console.print("[bold cyan][Stage 1: NCM Concept & Structure][/bold cyan] Generating hierarchical draft on MPS...")

        runner = NCMRunner(task=task)
        t0 = time.perf_counter()
        res = runner.generate_coarse_to_fine(raw_input)
        t_ncm = time.perf_counter() - t0

        draft_text = res["text"]
        nfe_stage1 = res["nfe"]
        levels = res["intermediate_texts"]

        self.console.print(f"  ✓ NCM Draft generated in [bold green]{t_ncm:.2f}s[/bold green] across [yellow]{nfe_stage1}[/yellow] hierarchical passes.")
        self.console.print(f"  [dim]Draft:[/dim] {draft_text[:120]}...\n")

        # Crucial for Phase 4: Free MPS VRAM before loading Stage 2!
        runner.cleanup()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return {
            "draft_text": draft_text,
            "nfe": nfe_stage1,
            "time_s": t_ncm,
            "levels": levels,
            "generated_tokens": res["generated_tokens"]
        }

    def run_stage2_mlx(self, original_input: str, ncm_draft: str, task: str = "xsum") -> Dict[str, any]:
        """Stage 2: Parallel Prefill & Polish via MLX on Apple Silicon Metal."""
        self.console.print("[bold cyan][Stage 2: Parallel Prefill & Polish][/bold cyan] Ingesting NCM draft into MLX...")
        import mlx_lm

        # Attempt primary model, fallback if network/disk constrained
        model, tokenizer = None, None
        active_model_id = self.mlx_model_id

        try:
            self.console.print(f"  Loading MLX model: [yellow]{active_model_id}[/yellow]...")
            model, tokenizer = mlx_lm.load(active_model_id)
        except Exception as e:
            self.console.print(f"  [yellow]Primary model {active_model_id} load failed ({e}). Falling back to {self.fallback_mlx_model_id}...[/yellow]")
            active_model_id = self.fallback_mlx_model_id
            model, tokenizer = mlx_lm.load(active_model_id)

        prompt = (
            f"<|im_start|>system\nYou are an expert assistant. Synthesize and finalize the concise summary based on the provided document and high-level concept draft.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"SOURCE DOCUMENT:\n{original_input[:800]}\n\n"
            f"STRUCTURAL NCM CONCEPT DRAFT:\n{ncm_draft}\n\n"
            f"Please verify, refine, and formulate the final polished summary.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        t0 = time.perf_counter()
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=128,
            verbose=False
        )
        t_mlx = time.perf_counter() - t0

        prompt_tokens = len(tokenizer.encode(prompt))
        out_tokens = len(tokenizer.encode(response))

        self.console.print(f"  ✓ MLX Stage 2 complete in [bold green]{t_mlx:.2f}s[/bold green] (Prefill: {prompt_tokens} tokens, Polish: {out_tokens} tokens).")
        self.console.print(f"  [dim]Final Polish:[/dim] {response.strip()[:140]}...\n")
        self.console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]\n")

        # Cleanup MLX
        del model
        del tokenizer
        gc.collect()

        return {
            "final_text": response.strip(),
            "model_id": active_model_id,
            "prefill_tokens": prompt_tokens,
            "polish_tokens": out_tokens,
            "time_s": t_mlx
        }

    def run_hybrid_interleaved(self, raw_input: Union[str, list], original_text: str, task: str = "xsum") -> Dict[str, any]:
        t_total_start = time.perf_counter()

        # Step 1: NCM
        s1 = self.run_stage1_ncm(raw_input, task=task)

        # Step 2: MLX
        s2 = self.run_stage2_mlx(original_text, s1["draft_text"], task=task)

        t_total = time.perf_counter() - t_total_start
        total_nfe = s1["nfe"] + s2["polish_tokens"]
        total_tokens = s1["generated_tokens"] + s2["polish_tokens"]

        return {
            "ncm_draft": s1["draft_text"],
            "final_text": s2["final_text"],
            "stage1_nfe": s1["nfe"],
            "stage2_prefill_tokens": s2["prefill_tokens"],
            "stage2_polish_tokens": s2["polish_tokens"],
            "total_nfe": total_nfe,
            "total_tokens": total_tokens,
            "total_time_s": t_total,
            "stage1_time_s": s1["time_s"],
            "stage2_time_s": s2["time_s"],
        }

import gc
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Path injection for external/ncm
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM_DIR = os.path.join(ROOT_DIR, "external", "ncm")
if NCM_DIR not in sys.path:
    sys.path.insert(0, NCM_DIR)

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer


# MPS compatibility patch for adaptive pooling in F.interpolate(mode='area')
_ORIG_INTERPOLATE = F.interpolate


def _mps_safe_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                          recompute_scale_factor=None, antialias=False):
    if input.device.type == "mps" and mode == "area":
        # Adaptive pooling on MPS fails when dimensions are non-divisible; compute on CPU and move back to MPS
        res = _ORIG_INTERPOLATE(
            input.cpu(), size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias
        )
        return res.to(input.device)
    return _ORIG_INTERPOLATE(
        input, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias
    )


F.interpolate = _mps_safe_interpolate


def get_apple_silicon_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NCMRunner:
    def __init__(
        self,
        task: str = "xsum",
        checkpoint_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.task = task
        self.device = device or get_apple_silicon_device()
        self.dtype = dtype

        if checkpoint_dir is None:
            if task in ("sudoku-hard", "sudoku/hard"):
                checkpoint_dir = os.path.join(ROOT_DIR, "artifacts", "sudoku-hard")
            else:
                checkpoint_dir = os.path.join(ROOT_DIR, "artifacts", task)

        if config_path is None:
            config_path = os.path.join(checkpoint_dir, "config.yaml")

        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path

        self._load_model()

    def _load_model(self):
        cfg = OmegaConf.load(self.config_path)
        self.cfg = cfg
        # Enforce SDPA on Apple Silicon MPS rather than CUDA flex_attention
        self.cfg.model.use_flex_attention = False
        self.cfg.tokenizer.vqvae.checkpoint_path = os.path.join(self.checkpoint_dir, cfg.checkpoints.tokenizer)

        # Load Tokenizer (pre-tokenizer + VQ-VAE)
        self.tokenizer = VQVAETokenizer.from_config(self.cfg.tokenizer)
        self.tokenizer.vqvae = self.tokenizer.vqvae.to(self.device).eval()

        # Load Next-Concept Generator Transformer
        self.net = NextConceptTransformer(vqvae_local=self.tokenizer.vqvae, config=self.cfg.model)
        gen_ckpt_path = os.path.join(self.checkpoint_dir, self.cfg.checkpoints.generator)
        sd = torch.load(gen_ckpt_path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(sd["model"], strict=True)
        self.net = self.net.to(self.device).eval()

        # Calculate parameters and size in GB
        self.vqvae_params = sum(p.numel() for p in self.tokenizer.vqvae.parameters())
        self.net_params = sum(p.numel() for p in self.net.parameters())
        self.total_params = self.vqvae_params + self.net_params
        self.model_params_m = self.total_params / 1e6

        # Element size (float32 = 4 bytes, float16 = 2 bytes)
        bytes_per_param = 4 if self.dtype == torch.float32 else 2
        self.model_size_gb = (self.total_params * bytes_per_param) / (1024 ** 3)

        self.levels = list(self.tokenizer.vqvae.vq.levels)
        self.n_levels = len(self.levels)
        self.prefix_len = int(self.cfg.model.prefix_len)

    def prepare_prefix_tensor(self, raw_input: Union[str, list, torch.Tensor]) -> torch.Tensor:
        """Encodes string or processes raw token list into a padded prefix tensor."""
        PAD = getattr(self.tokenizer.pre_tokenizer, "pad_token_id", None)
        if PAD is None:
            PAD = 0

        if isinstance(raw_input, torch.Tensor):
            ids = raw_input.tolist()
        elif isinstance(raw_input, str):
            if hasattr(self.tokenizer.pre_tokenizer, "encode"):
                ids = self.tokenizer.pre_tokenizer.encode(raw_input)
            else:
                ids = list(raw_input.encode("utf-8"))
        elif isinstance(raw_input, list):
            ids = list(raw_input)
        else:
            raise TypeError(f"Unsupported input type: {type(raw_input)}")

        padded = (ids[:self.prefix_len] + [PAD] * self.prefix_len)[:self.prefix_len]
        return torch.tensor([padded], device=self.device, dtype=torch.long)

    def decode_tokens(self, token_ids: List[int]) -> str:
        EOS = getattr(self.tokenizer.pre_tokenizer, "eos_token_id", None)
        PAD = getattr(self.tokenizer.pre_tokenizer, "pad_token_id", None)
        filtered = [t for t in token_ids if t not in (EOS, PAD)]
        if hasattr(self.tokenizer.pre_tokenizer, "decode"):
            return self.tokenizer.pre_tokenizer.decode(filtered, skip_special_tokens=True)
        return str(filtered)

    @torch.no_grad()
    def generate_coarse_to_fine(
        self,
        raw_input: Union[str, list, torch.Tensor],
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.0,
        cfg_scale: float = 0.0,
        on_level_callback: Optional[Callable[[int, int, str, int], None]] = None
    ) -> Dict[str, any]:
        """
        Executes NCM coarse-to-fine generation across all hierarchy levels.
        Invokes on_level_callback(level_idx, total_levels, intermediate_text, pos_count) after each level.
        """
        x = self.prepare_prefix_tensor(raw_input)
        t_start = time.perf_counter()

        # Run hierarchical generation passes
        g = self.net.generate(
            None, x,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale
        )
        if str(self.device) == "mps" or (hasattr(self.device, "type") and self.device.type == "mps"):
            torch.mps.synchronize()
        t_gen_done = time.perf_counter()

        # Extract multi-scale latent z levels
        _, z_levels = self.tokenizer.vqvae.vq.idx_to_zhats(g, return_zlevels=True)

        intermediate_texts = []
        for lvl_idx, z in enumerate(z_levels):
            logits = self.tokenizer.vqvae.decoder(z)
            pred_tokens = logits.argmax(-1).tolist()[0]
            lvl_text = self.decode_tokens(pred_tokens)
            intermediate_texts.append(lvl_text)
            if on_level_callback is not None:
                on_level_callback(lvl_idx, len(z_levels), lvl_text, z.shape[1])

        if str(self.device) == "mps" or (hasattr(self.device, "type") and self.device.type == "mps"):
            torch.mps.synchronize()
        t_decode_done = time.perf_counter()

        final_text = intermediate_texts[-1] if intermediate_texts else ""
        final_token_count = self.levels[-1]

        return {
            "text": final_text,
            "intermediate_texts": intermediate_texts,
            "nfe": self.n_levels,
            "generated_tokens": final_token_count,
            "generation_time_s": t_gen_done - t_start,
            "decode_time_s": t_decode_done - t_gen_done,
            "total_time_s": t_decode_done - t_start,
            "tokens_per_sec": final_token_count / max(t_decode_done - t_start, 1e-6),
        }

    def cleanup(self):
        del self.net
        del self.tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

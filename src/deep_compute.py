import gc
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from rich.console import Console
from rouge_score import rouge_scorer

from src.ncm_mps_runner import NCMRunner
from src.telemetry import TelemetryContext, get_peak_memory_gb


def check_sudoku_validity(grid_str: str) -> Tuple[int, int, bool]:
    """
    Checks how many Sudoku constraints (rows, cols, 3x3 boxes) are satisfied.
    Total constraints: 9 rows + 9 cols + 9 boxes = 27.
    Returns (satisfied_constraints, total_constraints, is_perfect).
    """
    # Clean characters
    chars = [c for c in grid_str if c in "123456789_"]
    if len(chars) < 81:
        chars = chars + ["0"] * (81 - len(chars))
    grid = [[int(c) if c in "123456789" else 0 for c in chars[i * 9:(i + 1) * 9]] for i in range(9)]

    satisfied = 0
    # Check rows
    for r in range(9):
        row = [grid[r][c] for c in range(9) if grid[r][c] != 0]
        if len(row) == 9 and len(set(row)) == 9:
            satisfied += 1

    # Check cols
    for c in range(9):
        col = [grid[r][c] for r in range(9) if grid[r][c] != 0]
        if len(col) == 9 and len(set(col)) == 9:
            satisfied += 1

    # Check 3x3 boxes
    for br in range(3):
        for bc in range(3):
            box = [grid[br * 3 + r][bc * 3 + c] for r in range(3) for c in range(3) if grid[br * 3 + r][bc * 3 + c] != 0]
            if len(box) == 9 and len(set(box)) == 9:
                satisfied += 1

    return satisfied, 27, (satisfied == 27)


def compute_mbr_consensus(candidates: List[str]) -> Tuple[int, str, List[float]]:
    """
    Minimum Bayes Risk (MBR) consensus decoding over candidates using ROUGE-L.
    Selects candidate with the highest average pairwise similarity to all others.
    """
    if len(candidates) == 1:
        return 0, candidates[0], [1.0]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    n = len(candidates)
    scores = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i != j:
                sim = scorer.score(candidates[i], candidates[j])["rougeL"].fmeasure
                scores[i] += sim
        scores[i] /= (n - 1)

    best_idx = int(torch.tensor(scores).argmax().item())
    return best_idx, candidates[best_idx], scores


class DeepComputeRunner:
    EFFORT_MAP = {
        "low": 4,       # 4 candidates
        "medium": 8,    # 8 candidates
        "high": 16,     # 16 candidates
        "extreme": 32   # 32 candidates
    }

    def __init__(self, runner: NCMRunner, max_ram_gb: float = 12.0, console: Optional[Console] = None):
        self.runner = runner
        self.max_ram_gb = max_ram_gb
        self.console = console or Console()

    def run_deep_compute(
        self,
        raw_input: Union[str, list],
        effort: str = "medium",
        cfg_scale: float = 1.0,
        task_type: str = "xsum"
    ) -> Dict[str, any]:
        """
        Executes Deep-Compute Test-Time Scaling:
        - Keeps model small in RAM (<12 GB budget strictly maintained).
        - Computes K candidates sequentially or micro-batched.
        - Applies MBR consensus (for text/xsum) or Rule-Constraint Refinement (for sudoku).
        - Forces memory cleanup after each candidate.
        """
        num_candidates = self.EFFORT_MAP.get(effort.lower(), 8)
        self.console.print(f"[bold cyan][Deep-Compute][/bold cyan] Effort: [bold yellow]{effort.upper()}[/bold yellow] ({num_candidates} Candidates, CFG: {cfg_scale}, MAX_RAM: {self.max_ram_gb} GB)")

        candidates: List[str] = []
        candidate_meta: List[Dict[str, any]] = []
        total_nfe = 0
        t0 = time.perf_counter()

        for k in range(num_candidates):
            # Check RAM limit before running
            cur_ram = get_peak_memory_gb()
            if cur_ram > self.max_ram_gb:
                self.console.print(f"[bold red]WARNING: RAM threshold exceeded ({cur_ram:.2f} GB > {self.max_ram_gb} GB). Aborting further candidates.[/bold red]")
                break

            # Slightly vary temperature & top_p to explore concept space
            temp = 0.7 + (k % 4) * 0.15
            top_p = 0.90 + (k % 3) * 0.04 if k > 0 else 0.0
            # Bound top_k to model vocab size (e.g. Sudoku vocab is 16, XSum is 4096)
            vocab_limit = 50
            if hasattr(self.runner.cfg.model, "vocab_size"):
                vs = self.runner.cfg.model.vocab_size
                try:
                    vocab_limit = min(list(vs))
                except Exception:
                    vocab_limit = int(vs)
            top_k = min(50, max(1, int(vocab_limit) - 1)) if k > 0 else 1

            # Only use CFG if the model has prefix_pad_id defined (e.g. XSum, not fixed-length Sudoku)
            has_pad_id = getattr(self.runner.net, "prefix_pad_id", None) is not None
            active_cfg = cfg_scale if (k % 2 == 1 and has_pad_id) else 0.0

            t_cand_start = time.perf_counter()
            res = self.runner.generate_coarse_to_fine(
                raw_input,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=active_cfg
            )
            cand_time = time.perf_counter() - t_cand_start
            total_nfe += res["nfe"]
            text = res["text"]
            candidates.append(text)

            meta = {
                "index": k + 1,
                "temp": temp,
                "top_p": top_p,
                "time_s": cand_time,
                "text": text
            }
            if task_type in ("sudoku", "sudoku-hard"):
                sat, total_c, is_perf = check_sudoku_validity(text)
                meta["constraints"] = f"{sat}/{total_c}"
                meta["is_perfect"] = is_perf

            candidate_meta.append(meta)
            self.console.print(f"  [cyan]Candidate {k+1:02d}/{num_candidates:02d}[/cyan] ({cand_time:.2f}s, T={temp:.2f}, RAM={cur_ram:.2f} GB): {text[:65]}...")

            # Clean memory caches after every candidate
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        t_total = time.perf_counter() - t0

        # Consensus selection
        if task_type in ("sudoku", "sudoku-hard"):
            # Select candidate with highest constraints satisfied
            best_idx = 0
            best_score = -1
            for idx, meta in enumerate(candidate_meta):
                sat = int(meta.get("constraints", "0/27").split("/")[0])
                if sat > best_score:
                    best_score = sat
                    best_idx = idx
            best_text = candidates[best_idx]
            selection_method = f"Sudoku Constraint Maximization ({best_score}/27 valid)"
        else:
            best_idx, best_text, scores = compute_mbr_consensus(candidates)
            selection_method = f"MBR ROUGE-L Consensus (Score: {scores[best_idx]:.3f})"

        raw_tok_s = (self.runner.levels[-1] * len(candidates)) / max(t_total, 1e-6)
        eff_tok_s = self.runner.levels[-1] / max(t_total, 1e-6)
        self.console.print(f"[bold green]✓ Best Candidate Selected:[/bold green] #{best_idx+1} via {selection_method}")
        self.console.print(f"  [dim]Aggregated Throughput across {len(candidates)} candidates: {raw_tok_s:.1f} tok/s | Effective Output Latency Throughput: {eff_tok_s:.1f} tok/s[/dim]")

        return {
            "best_text": best_text,
            "best_index": best_idx + 1,
            "selection_method": selection_method,
            "candidates": candidates,
            "candidate_meta": candidate_meta,
            "num_candidates": len(candidates),
            "total_nfe": total_nfe,
            "total_time_s": t_total,
            "tokens_per_sec": raw_tok_s,
            "effective_tok_s": eff_tok_s
        }

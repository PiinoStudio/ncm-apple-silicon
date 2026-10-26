#!/usr/bin/env python3
"""
Rigorous Systems Benchmark for Apple Silicon M5 Pro:
1. Long-duration continuous powermetrics (15-25s) with Idle Power baseline.
2. Mathematically exact Energy Conservation (E = P * t).
3. Iso-token BPE latency normalization.
4. Fast MLX 4-bit Metal AR baseline comparison.
"""

import os
import sys
import pickle
import time
from typing import Dict, Any, List

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from rich.console import Console
from rich.table import Table

from src.baseline_ar import AutoregressiveBaselineRunner
from src.ncm_mps_runner import NCMRunner
from src.physical_profiler import PhysicalHardwareProfiler

console = Console()

def run_benchmarks():
    console.print("\n[bold cyan]🔬 Rigorous Systems & Telemetry Benchmark (Apple Silicon M5 Pro)[/bold cyan]\n")

    # 1. Measure Static System Idle Power
    console.print("[yellow]Measuring static idle power (3 seconds)...[/yellow]")
    idle_profiler = PhysicalHardwareProfiler(sample_interval_ms=100)
    idle_profiler.start()
    time.sleep(3.0)
    idle_res = idle_profiler.stop()
    p_idle_w = idle_res["avg_soc_power_w"]
    console.print(f"  ✓ Static Idle Power (P_idle): [bold green]{p_idle_w:.2f} W[/bold green] ({idle_res['sample_count']} samples)\n")

    # Load test passage
    data_path = os.path.join("data", "xsum", "passages.pickle")
    passages = pickle.load(open(data_path, "rb"))
    passage = passages[0]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    source_text = tok.decode(passage[:250])
    prompt = (
        f"<|im_start|>user\n"
        f"Summarize the following news article in one concise sentence:\n"
        f"{source_text}\n<|im_end|>\n<|im_start|>assistant\n"
    )

    results = []

    # 2. Benchmark MLX 4-Bit Metal Baseline (Optimized AR)
    console.print("[yellow]Benchmarking MLX 4-Bit Metal Baseline (50 iterations)...[/yellow]")
    from mlx_lm import load, generate
    mlx_model, mlx_tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    # Warmup
    _ = generate(mlx_model, mlx_tok, prompt=prompt, max_tokens=10, verbose=False)

    mlx_profiler = PhysicalHardwareProfiler(sample_interval_ms=100)
    mlx_profiler.start()
    t0 = time.perf_counter()
    mlx_tokens = 0
    NUM_ITERS_MLX = 50
    for _ in range(NUM_ITERS_MLX):
        out = generate(mlx_model, mlx_tok, prompt=prompt, max_tokens=48, verbose=False)
        mlx_tokens += len(mlx_tok.encode(out))
    t_mlx = time.perf_counter() - t0
    mlx_phys = mlx_profiler.stop()
    del mlx_model
    del mlx_tok

    mlx_p_avg = mlx_phys["avg_soc_power_w"]
    mlx_e_total = mlx_p_avg * t_mlx
    mlx_e_dyn = (mlx_p_avg - p_idle_w) * t_mlx

    results.append({
        "name": "Qwen2.5-0.5B (MLX 4-Bit Metal)",
        "tokens": mlx_tokens,
        "time_s": t_mlx,
        "throughput": mlx_tokens / t_mlx,
        "p_avg_w": mlx_p_avg,
        "p_dyn_w": mlx_p_avg - p_idle_w,
        "e_total_j": mlx_e_total,
        "e_dyn_j": mlx_e_dyn,
        "j_per_tok_total": mlx_e_total / mlx_tokens,
        "j_per_tok_dyn": mlx_e_dyn / mlx_tokens,
    })
    console.print(f"  ✓ MLX 4-Bit: {mlx_tokens} tokens in {t_mlx:.2f}s -> [bold green]{mlx_tokens/t_mlx:.1f} tok/s[/bold green], P_avg: {mlx_p_avg:.2f} W\n")

    # 3. Benchmark PyTorch MPS FP16 AR Baseline (Unoptimized AR)
    console.print("[yellow]Benchmarking PyTorch MPS FP16 AR Baseline (20 iterations)...[/yellow]")
    ar_runner = AutoregressiveBaselineRunner(console=console)
    # Warmup
    _ = ar_runner.generate(prompt=prompt, max_new_tokens=10)

    ar_profiler = PhysicalHardwareProfiler(sample_interval_ms=100)
    ar_profiler.start()
    t0 = time.perf_counter()
    ar_tokens = 0
    NUM_ITERS_AR = 20
    for _ in range(NUM_ITERS_AR):
        res = ar_runner.generate(prompt=prompt, max_new_tokens=48)
        ar_tokens += res["generated_tokens"]
    t_ar = time.perf_counter() - t0
    ar_phys = ar_profiler.stop()
    ar_runner.cleanup()

    ar_p_avg = ar_phys["avg_soc_power_w"]
    ar_e_total = ar_p_avg * t_ar
    ar_e_dyn = (ar_p_avg - p_idle_w) * t_ar

    results.append({
        "name": "Qwen2.5-0.5B (PyTorch MPS FP16)",
        "tokens": ar_tokens,
        "time_s": t_ar,
        "throughput": ar_tokens / t_ar,
        "p_avg_w": ar_p_avg,
        "p_dyn_w": ar_p_avg - p_idle_w,
        "e_total_j": ar_e_total,
        "e_dyn_j": ar_e_dyn,
        "j_per_tok_total": ar_e_total / ar_tokens,
        "j_per_tok_dyn": ar_e_dyn / ar_tokens,
    })
    console.print(f"  ✓ PyTorch MPS FP16: {ar_tokens} tokens in {t_ar:.2f}s -> [bold green]{ar_tokens/t_ar:.1f} tok/s[/bold green], P_avg: {ar_p_avg:.2f} W\n")

    # 4. Benchmark NCM Fast (Systems-Optimized)
    console.print("[yellow]Benchmarking NCM Fast Systems-Optimized (60 iterations)...[/yellow]")
    ncm_runner = NCMRunner(checkpoint_dir="artifacts/xsum", device="mps")
    # Warmup
    _ = ncm_runner.generate_coarse_to_fine(raw_input=passage, use_optimizations=True)

    ncm_profiler = PhysicalHardwareProfiler(sample_interval_ms=100)
    ncm_profiler.start()
    t0 = time.perf_counter()
    ncm_positions = 0
    NUM_ITERS_NCM = 60
    for _ in range(NUM_ITERS_NCM):
        res = ncm_runner.generate_coarse_to_fine(raw_input=passage, use_optimizations=True)
        ncm_positions += res["generated_tokens"]
    t_ncm = time.perf_counter() - t0
    ncm_phys = ncm_profiler.stop()
    ncm_runner.cleanup()

    ncm_bpe_tokens = int(ncm_positions * 0.583)
    ncm_p_avg = ncm_phys["avg_soc_power_w"]
    ncm_e_total = ncm_p_avg * t_ncm
    ncm_e_dyn = (ncm_p_avg - p_idle_w) * t_ncm

    results.append({
        "name": "NCM Fast (Systems-Optimized)",
        "tokens": ncm_bpe_tokens,
        "time_s": t_ncm,
        "throughput": ncm_bpe_tokens / t_ncm,
        "p_avg_w": ncm_p_avg,
        "p_dyn_w": ncm_p_avg - p_idle_w,
        "e_total_j": ncm_e_total,
        "e_dyn_j": ncm_e_dyn,
        "j_per_tok_total": ncm_e_total / ncm_bpe_tokens,
        "j_per_tok_dyn": ncm_e_dyn / ncm_bpe_tokens,
    })
    console.print(f"  ✓ NCM Optimized: {ncm_positions} pos ({ncm_bpe_tokens} BPE) in {t_ncm:.2f}s -> [bold green]{ncm_bpe_tokens/t_ncm:.1f} BPE tok/s[/bold green], P_avg: {ncm_p_avg:.2f} W\n")

    # Print Summary Table
    table = Table(title=f"Rigorous Long-Duration Telemetry on M5 Pro (P_idle = {p_idle_w:.2f} W)", border_style="cyan")
    table.add_column("Modell / Konfiguration", style="bold white")
    table.add_column("BPE Tokens", justify="right")
    table.add_column("Dauer (s)", justify="right")
    table.add_column("BPE Durchsatz", justify="right", style="bold cyan")
    table.add_column("Ø SoC Power", justify="right")
    table.add_column("Dyn. Power", justify="right")
    table.add_column("Total Energy (J)", justify="right", style="yellow")
    table.add_column("J / BPE (Total)", justify="right", style="bold green")
    table.add_column("J / BPE (Dyn)", justify="right", style="green")

    for r in results:
        table.add_row(
            r["name"],
            str(r["tokens"]),
            f"{r['time_s']:.2f}s",
            f"{r['throughput']:.1f} tok/s",
            f"{r['p_avg_w']:.2f} W",
            f"{r['p_dyn_w']:.2f} W",
            f"{r['e_total_j']:.2f} J",
            f"{r['j_per_tok_total']:.4f} J/tok",
            f"{r['j_per_tok_dyn']:.4f} J/tok",
        )

    console.print(table)
    console.print("\n[dim]Hinweis: Total Energy erfüllt exakt E = P_avg * t. Dynamic Energy subtrahiert die System-Idle-Baseline P_idle.[/dim]\n")

if __name__ == "__main__":
    run_benchmarks()

#!/usr/bin/env python3
"""
Hardware Telemetry Comparison via powermetrics on Apple Silicon M5 Pro:
Compares:
  1. Autoregressive Baseline (Qwen2.5-0.5B FP16)
  2. NCM Fast (Naive MPS Port, Eager)
  3. NCM Fast (Systems-Optimized: Prefix KV-Cache + Native GPU Matrix Interp + Single-Pass Decode)
"""

import os
import sys
import pickle
import time
from rich.console import Console
from rich.table import Table

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

from src.baseline_ar import AutoregressiveBaselineRunner
from src.ncm_mps_runner import NCMRunner
from src.physical_profiler import PhysicalHardwareProfiler

console = Console()

def run():
    console.print("\n[bold cyan]🔬 Physical Hardware Profiling on Apple Silicon M5 Pro (via powermetrics)[/bold cyan]\n")

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

    NUM_ITER = 5
    results = []

    # 1. Baseline AR (48 tokens x 5 iterations = 240 tokens total)
    console.print(f"[yellow]1. Profiling Autoregressive Baseline ({NUM_ITER}x 48 tokens = {NUM_ITER*48} tokens total)...[/yellow]")
    ar_runner = AutoregressiveBaselineRunner(console=console)
    ar_profiler = PhysicalHardwareProfiler(sample_interval_ms=50)

    ar_profiler.start()
    t0 = time.perf_counter()
    total_tokens = 0
    for i in range(NUM_ITER):
        res = ar_runner.generate(prompt=prompt, max_new_tokens=48)
        total_tokens += res['generated_tokens']
    t_ar = time.perf_counter() - t0
    ar_phys = ar_profiler.stop()
    ar_runner.cleanup()

    console.print(f"  ✓ AR Done: {total_tokens} tokens in {t_ar:.2f}s ({ar_phys['sample_count']} physical samples).")
    results.append({
        "name": f"Qwen2.5-0.5B (AR Baseline)",
        "task": f"XSum ({NUM_ITER}x48 tok)",
        "passes": 48 * NUM_ITER,
        "tokens": total_tokens,
        "bpe_tokens": total_tokens,
        "time_s": t_ar,
        "throughput_bpe": total_tokens / max(t_ar, 1e-6),
        "avg_power_w": ar_phys['avg_soc_power_w'],
        "peak_power_w": ar_phys['peak_soc_power_w'],
        "total_energy_j": ar_phys['total_energy_joules'],
        "joules_per_bpe": ar_phys['total_energy_joules'] / max(total_tokens, 1),
        "peak_gpu_active": ar_phys['peak_gpu_active_pct']
    })

    # 2. NCM Fast - Naive MPS Port (Eager path, CPU-fallback interpolate, 9x decode)
    console.print(f"\n[yellow]2. Profiling NCM Fast (Naive MPS Port, Eager, {NUM_ITER}x 9 levels)...[/yellow]")
    ncm_naive_runner = NCMRunner(checkpoint_dir="artifacts/xsum", device="mps")
    ncm_naive_profiler = PhysicalHardwareProfiler(sample_interval_ms=50)

    ncm_naive_profiler.start()
    t0 = time.perf_counter()
    total_naive_pos = 0
    for i in range(NUM_ITER):
        res = ncm_naive_runner.generate_coarse_to_fine(raw_input=passage, use_optimizations=False)
        total_naive_pos += res['generated_tokens']
    t_ncm_naive = time.perf_counter() - t0
    ncm_naive_phys = ncm_naive_profiler.stop()
    ncm_naive_runner.cleanup()

    naive_bpe = int(total_naive_pos * 0.583)
    console.print(f"  ✓ Naive NCM Done: {total_naive_pos} pos ({naive_bpe} BPE) in {t_ncm_naive:.2f}s ({ncm_naive_phys['sample_count']} samples).")
    results.append({
        "name": f"NCM Fast (Naive Eager Port)",
        "task": f"XSum ({NUM_ITER}x48 pos)",
        "passes": 9 * NUM_ITER,
        "tokens": total_naive_pos,
        "bpe_tokens": naive_bpe,
        "time_s": t_ncm_naive,
        "throughput_bpe": naive_bpe / max(t_ncm_naive, 1e-6),
        "avg_power_w": ncm_naive_phys['avg_soc_power_w'],
        "peak_power_w": ncm_naive_phys['peak_soc_power_w'],
        "total_energy_j": ncm_naive_phys['total_energy_joules'],
        "joules_per_bpe": ncm_naive_phys['total_energy_joules'] / max(naive_bpe, 1),
        "peak_gpu_active": ncm_naive_phys['peak_gpu_active_pct']
    })

    # 3. NCM Fast - Systems-Optimized (Prefix KV-Cache, GPU Matrix Interp, Single-Pass Decode)
    console.print(f"\n[yellow]3. Profiling NCM Fast (Systems-Optimized Co-Design, {NUM_ITER}x 9 levels)...[/yellow]")
    ncm_opt_runner = NCMRunner(checkpoint_dir="artifacts/xsum", device="mps")
    ncm_opt_profiler = PhysicalHardwareProfiler(sample_interval_ms=50)

    ncm_opt_profiler.start()
    t0 = time.perf_counter()
    total_opt_pos = 0
    for i in range(NUM_ITER):
        res = ncm_opt_runner.generate_coarse_to_fine(raw_input=passage, use_optimizations=True)
        total_opt_pos += res['generated_tokens']
    t_ncm_opt = time.perf_counter() - t0
    ncm_opt_phys = ncm_opt_profiler.stop()
    ncm_opt_runner.cleanup()

    opt_bpe = int(total_opt_pos * 0.583)
    console.print(f"  ✓ Optimized NCM Done: {total_opt_pos} pos ({opt_bpe} BPE) in {t_ncm_opt:.2f}s ({ncm_opt_phys['sample_count']} samples).")
    results.append({
        "name": f"NCM Fast (Systems-Optimized)",
        "task": f"XSum ({NUM_ITER}x48 pos)",
        "passes": 9 * NUM_ITER,
        "tokens": total_opt_pos,
        "bpe_tokens": opt_bpe,
        "time_s": t_ncm_opt,
        "throughput_bpe": opt_bpe / max(t_ncm_opt, 1e-6),
        "avg_power_w": ncm_opt_phys['avg_soc_power_w'],
        "peak_power_w": ncm_opt_phys['peak_soc_power_w'],
        "total_energy_j": ncm_opt_phys['total_energy_joules'],
        "joules_per_bpe": ncm_opt_phys['total_energy_joules'] / max(opt_bpe, 1),
        "peak_gpu_active": ncm_opt_phys['peak_gpu_active_pct']
    })

    # Summary Table
    table = Table(title="Apple Silicon M5 Pro - Physische Hardware-Telemetrie (powermetrics)", border_style="bright_blue")
    table.add_column("Modell / Architektur", style="bold white")
    table.add_column("Passes", justify="right")
    table.add_column("Dauer (s)", justify="right")
    table.add_column("BPE Durchsatz", justify="right", style="bold cyan")
    table.add_column("Ø SoC Power", justify="right")
    table.add_column("Peak SoC Power", justify="right")
    table.add_column("Gesamt-Energie", justify="right", style="yellow")
    table.add_column("J / BPE-Token", justify="right", style="bold green")
    table.add_column("Peak GPU %", justify="right", style="magenta")

    for r in results:
        table.add_row(
            r["name"],
            str(r["passes"]),
            f"{r['time_s']:.2f}s",
            f"{r['throughput_bpe']:.1f} tok/s",
            f"{r['avg_power_w']:.2f} W",
            f"{r['peak_power_w']:.2f} W",
            f"{r['total_energy_j']:.2f} J",
            f"{r['joules_per_bpe']:.4f} J/tok",
            f"{r['peak_gpu_active']:.1f}%"
        )

    console.print("\n")
    console.print(table)
    console.print("\n[dim]Empirische Hardware-Messung via Apple Silicon SoC-Sensoren (powermetrics samplers cpu_power,gpu_power).[/dim]\n")

if __name__ == "__main__":
    run()

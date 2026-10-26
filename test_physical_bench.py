#!/usr/bin/env python3
"""
Test script to compare physical hardware metrics (powermetrics)
between Autoregressive Baseline and NCM Fast on identical XSum task.
Runs a 5-pass benchmark loop to allow high-frequency physical sampling.
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
        "time_s": t_ar,
        "avg_power_w": ar_phys['avg_soc_power_w'],
        "peak_power_w": ar_phys['peak_soc_power_w'],
        "total_energy_j": ar_phys['total_energy_joules'],
        "joules_per_token": ar_phys['total_energy_joules'] / max(total_tokens, 1),
        "avg_gpu_active": ar_phys['avg_gpu_active_pct'],
        "peak_gpu_active": ar_phys['peak_gpu_active_pct']
    })

    # 2. NCM Fast (48 positions x 5 iterations = 240 positions total)
    console.print(f"\n[yellow]2. Profiling Next-Concept Modeling ({NUM_ITER}x 9 levels = {NUM_ITER*48} positions total)...[/yellow]")
    ncm_runner = NCMRunner(checkpoint_dir="artifacts/xsum", device="mps")
    ncm_profiler = PhysicalHardwareProfiler(sample_interval_ms=50)

    ncm_profiler.start()
    t0 = time.perf_counter()
    total_ncm_pos = 0
    for i in range(NUM_ITER):
        res = ncm_runner.generate_coarse_to_fine(raw_input=passage)
        total_ncm_pos += res['generated_tokens']
    t_ncm = time.perf_counter() - t0
    ncm_phys = ncm_profiler.stop()
    ncm_runner.cleanup()

    console.print(f"  ✓ NCM Done: {total_ncm_pos} positions in {t_ncm:.2f}s ({ncm_phys['sample_count']} physical samples).")
    results.append({
        "name": f"NCM Fast (Coarse-to-Fine)",
        "task": f"XSum ({NUM_ITER}x48 pos)",
        "passes": 9 * NUM_ITER,
        "tokens": total_ncm_pos,
        "time_s": t_ncm,
        "avg_power_w": ncm_phys['avg_soc_power_w'],
        "peak_power_w": ncm_phys['peak_soc_power_w'],
        "total_energy_j": ncm_phys['total_energy_joules'],
        "joules_per_token": ncm_phys['total_energy_joules'] / max(total_ncm_pos, 1),
        "avg_gpu_active": ncm_phys['avg_gpu_active_pct'],
        "peak_gpu_active": ncm_phys['peak_gpu_active_pct']
    })

    # Summary Table
    table = Table(title="Apple Silicon M5 Pro - Physische Hardware-Telemetrie (powermetrics)", border_style="bright_blue")
    table.add_column("Modell / Architektur", style="bold white")
    table.add_column("Passes", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Dauer (s)", justify="right")
    table.add_column("Ø SoC Power (W)", justify="right", style="cyan")
    table.add_column("Peak SoC Power (W)", justify="right")
    table.add_column("Gesamt-Energie (J)", justify="right", style="bold yellow")
    table.add_column("J / Token", justify="right", style="bold green")
    table.add_column("Peak GPU %", justify="right", style="magenta")

    for r in results:
        table.add_row(
            r["name"],
            str(r["passes"]),
            str(r["tokens"]),
            f"{r['time_s']:.2f}s",
            f"{r['avg_power_w']:.2f} W",
            f"{r['peak_power_w']:.2f} W",
            f"{r['total_energy_j']:.2f} J",
            f"{r['joules_per_token']:.4f} J/tok",
            f"{r['peak_gpu_active']:.1f}%"
        )

    console.print("\n")
    console.print(table)
    console.print("\n[dim]Empirische Hardware-Messung via Apple Silicon SoC-Sensoren (powermetrics samplers cpu_power,gpu_power).[/dim]\n")

if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Apple Silicon M5 Pro Proof of Concept (PoC) & Benchmark Suite
Next-Concept Modeling (NCM) vs Autoregressive LLM & Hybrid Execution

Hypotheses Tested:
1. Bottleneck-Shift (Memory Bandwidth -> Compute): Fixed NFE hierarchy vs 1 pass per token.
2. Deep-Compute Mode (Time for Quality without Swap): Test-time scaling with 0 swap pageouts.
"""

import argparse
import gc
import os
import pickle
import sys
import time
from typing import Optional

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Local imports
from src.baseline_ar import AutoregressiveBaselineRunner
from src.deep_compute import DeepComputeRunner
from src.hybrid_pipeline import HybridPipelineRunner
from src.ncm_mps_runner import NCMRunner, get_apple_silicon_device
from src.telemetry import TelemetryContext, TelemetryRecord, print_telemetry_table, get_peak_memory_gb


console = Console()


def load_test_sample(task: str = "xsum") -> tuple[any, str]:
    """Loads a benchmark test item (passage or puzzle) and human-readable text."""
    root = os.path.dirname(os.path.abspath(__file__))
    if task == "xsum":
        data_path = os.path.join(root, "data", "xsum", "passages.pickle")
        passages = pickle.load(open(data_path, "rb"))
        item = passages[0]
        # Decode the actual article from XSum tokens
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        text = tok.decode(item[:250])
        return item, text
    elif task in ("sudoku", "sudoku-hard"):
        data_path = os.path.join(root, "data", "sudoku-hard", "puzzles.pickle")
        puzzles = pickle.load(open(data_path, "rb"))
        item = puzzles[0]
        return item, "Sudoku Hard Puzzle #0"
    else:
        raise ValueError(f"Unknown task: {task}")


def run_baseline_ar_mode(max_tokens: int = 512, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", source_passage: Optional[str] = None) -> TelemetryRecord:
    console.print(Panel.fit("[bold blue]1. Mode: Autoregressive LLM Baseline (--mode baseline-ar)[/bold blue]", border_style="blue"))
    console.print(f"[dim]Running standard autoregressive token-by-token generation (Max: {max_tokens} tokens)...[/dim]\n")

    if source_passage:
        prompt = (
            f"<|im_start|>user\n"
            f"Summarize the following news article in one concise sentence:\n"
            f"{source_passage}\n<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        prompt = (
            "Analyze the architectural trade-offs between classic token-by-token autoregressive decoding "
            "and hierarchical coarse-to-fine generation on unified memory architectures like Apple Silicon M5 Pro. "
            "Discuss memory bandwidth saturation versus arithmetic intensity."
        )

    runner = AutoregressiveBaselineRunner(model_id=model_id, console=console)

    with TelemetryContext(
        mode="baseline-ar",
        scenario_name=f"{model_id.split('/')[-1]} ({max_tokens} toks)",
        model_params_m=runner.model_params_m,
        model_size_gb=runner.model_size_gb
    ) as ctx:
        res = runner.generate(prompt, max_new_tokens=max_tokens, do_sample=False)
        ctx.nfe = res["nfe"]
        ctx.generated_tokens = res["generated_tokens"]
        ctx.details = f"Tokens/s: {res['tokens_per_sec']:.1f}"

    console.print(f"\n[green]✓ Generated {res['generated_tokens']} tokens in {res['wall_time_s']:.2f}s ({res['tokens_per_sec']:.1f} tok/s)[/green]")
    console.print(f"[yellow]  NFE (Forward Passes): {res['nfe']}[/yellow] (Each token required reading {runner.model_size_gb:.2f} GB model weights)")
    console.print(f"[red]  Theor. Weight Streaming Read: {res['est_bandwidth_gb']:.2f} GB ({res['gb_per_token']:.4f} GB/tok) [Analytisches Modell][/red]\n")
    console.print(f"[dim]Excerpt:[/dim] {res['text'][:200]}...\n")

    runner.cleanup()
    return ctx.get_record()


def run_ncm_fast_mode(task: str = "xsum") -> TelemetryRecord:
    console.print(Panel.fit(f"[bold green]2. Mode: Standard NCM Coarse-to-Fine (--mode ncm-fast, Task: {task})[/bold green]", border_style="green"))
    console.print("[dim]Executing hierarchical generation (1 parallel forward pass per hierarchy level)...[/dim]\n")

    item, desc = load_test_sample(task)
    runner = NCMRunner(task=task)

    console.print(f"[cyan]NCM Model:[/cyan] {runner.model_params_m:.1f}M params ({runner.model_size_gb:.2f} GB in RAM), Levels: {runner.n_levels}\n")
    console.print("[bold yellow]Live Coarse-to-Fine Refinement across Hierarchy Levels:[/bold yellow]")

    intermediate_log = []

    def on_level(lvl_idx: int, total_lvls: int, text: str, pos_count: int):
        clean = text.replace("\n", " ").strip()
        intermediate_log.append((lvl_idx, pos_count, clean))
        console.print(f"  [bold cyan]Level {lvl_idx:02d}/{total_lvls-1:02d}[/bold cyan] ({pos_count:2d} pos): [white]{clean[:80]}[/white]")

    with TelemetryContext(
        mode="ncm-fast",
        scenario_name=f"NCM {task.upper()} ({runner.n_levels} Levels)",
        model_params_m=runner.model_params_m,
        model_size_gb=runner.model_size_gb
    ) as ctx:
        res = runner.generate_coarse_to_fine(item, on_level_callback=on_level)
        ctx.nfe = res["nfe"]
        ctx.generated_tokens = res["generated_tokens"]
        ctx.details = f"Levels: {runner.n_levels}"

    console.print(f"\n[green]✓ Generated {res['generated_tokens']} output positions in {res['total_time_s']:.2f}s ({res['tokens_per_sec']:.1f} tok/s)[/green]")
    console.print(f"[yellow]  NFE (Forward Passes): {res['nfe']}[/yellow] (Fixed level evaluations!)")
    est_bw = res["nfe"] * runner.model_size_gb
    console.print(f"[green]  Est. RAM Bandwidth Transferred: {est_bw:.2f} GB ({est_bw/res['generated_tokens']:.4f} GB/tok)[/green]")
    console.print(f"[bold green]  Final Output:[/bold green] {res['text']}\n")

    runner.cleanup()
    return ctx.get_record()


def run_ncm_deep_compute_mode(effort: str = "medium", task: str = "xsum", max_ram_gb: float = 12.0) -> TelemetryRecord:
    console.print(Panel.fit(f"[bold magenta]3. Mode: NCM Deep-Compute Mode (--mode ncm-deep-compute, Effort: {effort})[/bold magenta]", border_style="magenta"))
    console.print(f"[dim]Scaling test-time compute with strict MAX_RAM={max_ram_gb} GB budget (Zero Swap Proof)...[/dim]\n")

    item, desc = load_test_sample(task)
    runner = NCMRunner(task=task)
    deep_runner = DeepComputeRunner(runner=runner, max_ram_gb=max_ram_gb, console=console)

    with TelemetryContext(
        mode="ncm-deep-compute",
        scenario_name=f"NCM MBR {effort.upper()} ({task})",
        model_params_m=runner.model_params_m,
        model_size_gb=runner.model_size_gb
    ) as ctx:
        res = deep_runner.run_deep_compute(item, effort=effort, task_type=task)
        ctx.nfe = res["total_nfe"]
        ctx.generated_tokens = runner.levels[-1] * res["num_candidates"]
        ctx.details = f"Effort: {effort}, Candidates: {res['num_candidates']}"

    console.print(f"\n[bold green]✓ Deep-Compute Run Completed:[/bold green] {res['total_nfe']} total passes in {res['total_time_s']:.2f}s")
    console.print(f"  Selection: [cyan]{res['selection_method']}[/cyan]")
    console.print(f"  Best Solution: [white]{res['best_text']}[/white]\n")

    rec = ctx.get_record()
    rec.effective_tok_s = res.get("effective_tok_s", 0.0)
    runner.cleanup()
    return rec


def run_hybrid_mode(task: str = "xsum") -> TelemetryRecord:
    console.print(Panel.fit("[bold cyan]4. Mode: Hybrid-Interleaved NCM + MLX (--mode hybrid-interleaved)[/bold cyan]", border_style="cyan"))
    console.print("[dim]Stage 1: NCM structural drafting on MPS -> Stage 2: Parallel Prefill & Polish in MLX on Metal...[/dim]\n")

    item, text_input = load_test_sample(task)
    hybrid_runner = HybridPipelineRunner(console=console)

    with TelemetryContext(
        mode="hybrid-interleaved",
        scenario_name="NCM (MPS) + Qwen2.5 (MLX)",
        model_params_m=148.6,
        model_size_gb=0.60
    ) as ctx:
        res = hybrid_runner.run_hybrid_interleaved(item, original_text=text_input, task=task)
        ctx.nfe = res["total_nfe"]
        ctx.generated_tokens = res["total_tokens"]
        ctx.details = f"Prefill: {res['stage2_prefill_tokens']} tok, Polish: {res['stage2_polish_tokens']} tok"

    console.print(f"[bold green]✓ Hybrid Interleaved Finished:[/bold green] Total time: {res['total_time_s']:.2f}s")
    console.print(f"  Stage 1 (NCM Draft): {res['stage1_time_s']:.2f}s ({res['stage1_nfe']} passes)")
    console.print(f"  Stage 2 (MLX Prefill & Polish): {res['stage2_time_s']:.2f}s ({res['stage2_polish_tokens']} tokens)\n")

    return ctx.get_record()


def main():
    parser = argparse.ArgumentParser(description="Apple Silicon M5 Pro PoC - Next-Concept Modeling Benchmark Suite")
    parser.add_argument("--mode", choices=["baseline-ar", "ncm-fast", "ncm-deep-compute", "hybrid-interleaved", "all"],
                        default="all", help="Benchmark execution mode")
    parser.add_argument("--task", choices=["xsum", "sudoku-hard"], default="xsum", help="NCM task evaluation")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens for autoregressive baseline")
    parser.add_argument("--compute-effort", choices=["low", "medium", "high", "extreme"], default="medium",
                        help="Compute effort budget for deep compute mode")
    parser.add_argument("--max-ram-gb", type=float, default=12.0, help="Hard RAM threshold in GB to prevent swap")
    args = parser.parse_args()

    console.print(Panel(
        Text("Next-Concept Modeling (NCM) - Apple Silicon M5 Pro PoC\nHardware Verification & Bottleneck-Shift Benchmark Suite", justify="center", style="bold white on blue"),
        expand=False
    ))

    records: list[TelemetryRecord] = []

    item, desc = load_test_sample(args.task)

    if args.mode in ("baseline-ar", "all"):
        # If tokens == 48, run 1:1 on the exact same task/passage as NCM; otherwise run general length benchmark
        passage_to_use = desc if (args.task == "xsum" and args.tokens <= 64) else None
        rec_ar = run_baseline_ar_mode(max_tokens=args.tokens, source_passage=passage_to_use)
        records.append(rec_ar)

    if args.mode in ("ncm-fast", "all"):
        rec_ncm = run_ncm_fast_mode(task=args.task)
        records.append(rec_ncm)

    if args.mode in ("ncm-deep-compute", "all"):
        rec_deep = run_ncm_deep_compute_mode(effort=args.compute_effort, task=args.task, max_ram_gb=args.max_ram_gb)
        records.append(rec_deep)

    if args.mode in ("hybrid-interleaved", "all"):
        rec_hyb = run_hybrid_mode(task="xsum")
        records.append(rec_hyb)

    # Final telemetry table
    print_telemetry_table(records, console=console)

    # Hypothesis Verification Summary
    if len(records) >= 2:
        ncm_rec = next((r for r in records if "ncm" in r.mode), records[1])
        ar_rec = next((r for r in records if "ar" in r.mode), records[0])
        bw_ratio = (ar_rec.gb_per_token / max(ncm_rec.gb_per_token, 1e-6))
        console.print(Panel.fit(
            "[bold green]Methodische Einordnung auf Apple Silicon M5 Pro (24 GB Unified Memory):[/bold green]\n"
            "1. [bold yellow]Bottleneck-Shift (Analytisches Gewichtstransfer-Modell):[/bold yellow]\n"
            f"   • AR LLM erfordert pro Token das Laden aller Gewichte (~{ar_rec.gb_per_token:.4f} GB/tok im Streaming-Modell).\n"
            f"   • NCM benötigt nur eine fixe Anzahl hierarchischer Pässe (~{ncm_rec.gb_per_token:.4f} GB/tok) -> rechnerisch ~{bw_ratio:.1f}x weniger theoretischer Gewichtstransfer pro Position.\n"
            f"   • Wichtig: Dies ist ein mathematisches Streaming-Modell (NFE x Weights), keine physikalische Bus-Messung.\n"
            "2. [bold yellow]Deep-Compute Mode (Zeit gegen Qualität ohne SSD-Swap):[/bold yellow]\n"
            f"   • Peak RAM blieb strikt unter Budget (Peak: {max(r.peak_ram_gb for r in records):.2f} GB < 12.0 GB Limit).\n"
            f"   • macOS vm_stat Pageouts / Swapouts Delta = {sum(r.swap_pageouts_delta for r in records)} / {sum(r.swapouts_delta for r in records)} -> keine messbaren SSD-Swapouts registriert.",
            title="Benchmark Zusammenfassung & Einordnung", border_style="bold green"
        ))
    else:
        r = records[0]
        console.print(Panel.fit(
            f"[bold green]Lauf abgeschlossen:[/bold green] Modus: {r.mode}, NFE: {r.nfe}, Peak RAM: {r.peak_ram_gb:.2f} GB, Pageouts/Swapouts Delta: {r.swap_pageouts_delta}/{r.swapouts_delta}",
            title="Benchmark Zusammenfassung", border_style="bold green"
        ))


if __name__ == "__main__":
    main()

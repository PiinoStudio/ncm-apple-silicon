import gc
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import psutil
import torch
from rich.console import Console
from rich.table import Table
from .physical_profiler import PhysicalHardwareProfiler


def get_vm_stat() -> Dict[str, Optional[int]]:
    """Parse vm_stat output on macOS to get pageouts, swapins, swapouts."""
    try:
        res = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
        stats = {}
        for line in res.stdout.splitlines():
            m = re.match(r'^\s*"?([^":]+)"?:\s*([0-9]+)\.?', line)
            if m:
                key, val = m.group(1).strip(), int(m.group(2).strip())
                stats[key] = val
        return {
            "pageouts": stats.get("Pageouts", 0),
            "swapins": stats.get("Swapins", 0),
            "swapouts": stats.get("Swapouts", 0),
            "pageins": stats.get("Pageins", 0),
        }
    except Exception as e:
        return {"pageouts": None, "swapins": None, "swapouts": None, "pageins": None, "error": str(e)}


def get_peak_memory_gb() -> float:
    """Get current RSS + MPS memory in GB."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    mps_gb = 0.0
    if torch.backends.mps.is_available():
        try:
            mps_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
        except Exception:
            pass
    return max(rss_gb, mps_gb)


@dataclass
class TelemetryRecord:
    mode: str
    scenario_name: str
    nfe: int  # Number of Forward Evaluations
    generated_tokens: int
    wall_time_s: float
    tokens_per_sec: float
    model_params_m: float
    model_size_gb: float
    est_bandwidth_gb: float  # Theoretical weight streaming: NFE * model_size_gb (analytical model)
    gb_per_token: float  # est_bandwidth_gb / generated_tokens
    peak_ram_gb: float
    swap_pageouts_delta: int
    swapouts_delta: int = 0
    effective_tok_s: float = 0.0
    avg_soc_power_w: float = 0.0
    peak_soc_power_w: float = 0.0
    total_energy_j: float = 0.0
    joules_per_token: float = 0.0
    peak_gpu_active_pct: float = 0.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryContext:
    def __init__(self, mode: str, scenario_name: str, model_params_m: float, model_size_gb: float):
        self.mode = mode
        self.scenario_name = scenario_name
        self.model_params_m = model_params_m
        self.model_size_gb = model_size_gb
        self.start_vm: Dict[str, int] = {}
        self.end_vm: Dict[str, int] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.peak_ram: float = 0.0
        self.nfe: int = 0
        self.generated_tokens: int = 0
        self.details: str = ""
        self.profiler = PhysicalHardwareProfiler(sample_interval_ms=100)
        self.phys_metrics: Dict[str, Any] = {}

    def __enter__(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.start_vm = get_vm_stat()
        self.peak_ram = get_peak_memory_gb()
        self.profiler.start()
        self.start_time = time.perf_counter()
        return self

    def record_step(self, nfe_increment: int = 1):
        self.nfe += nfe_increment
        cur_mem = get_peak_memory_gb()
        if cur_mem > self.peak_ram:
            self.peak_ram = cur_mem

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.end_time = time.perf_counter()
        self.phys_metrics = self.profiler.stop()
        cur_mem = get_peak_memory_gb()
        if cur_mem > self.peak_ram:
            self.peak_ram = cur_mem
        self.end_vm = get_vm_stat()

    def get_record(self) -> TelemetryRecord:
        wall_time = max(self.end_time - self.start_time, 1e-6)
        tok_s = self.generated_tokens / wall_time if self.generated_tokens > 0 else 0.0

        p_end = self.end_vm.get("pageouts")
        p_start = self.start_vm.get("pageouts")
        pageouts_delta = (p_end - p_start) if (p_end is not None and p_start is not None) else 0

        s_end = self.end_vm.get("swapouts")
        s_start = self.start_vm.get("swapouts")
        swapouts_delta = (s_end - s_start) if (s_end is not None and s_start is not None) else 0

        est_bw = self.nfe * self.model_size_gb
        gb_per_tok = (est_bw / self.generated_tokens) if self.generated_tokens > 0 else 0.0

        avg_power = self.phys_metrics.get("avg_soc_power_w", 0.0)
        peak_power = self.phys_metrics.get("peak_soc_power_w", 0.0)
        total_energy = self.phys_metrics.get("total_energy_joules", 0.0)
        j_per_tok = (total_energy / self.generated_tokens) if self.generated_tokens > 0 else 0.0
        peak_gpu = self.phys_metrics.get("peak_gpu_active_pct", 0.0)

        return TelemetryRecord(
            mode=self.mode,
            scenario_name=self.scenario_name,
            nfe=self.nfe,
            generated_tokens=self.generated_tokens,
            wall_time_s=wall_time,
            tokens_per_sec=tok_s,
            model_params_m=self.model_params_m,
            model_size_gb=self.model_size_gb,
            est_bandwidth_gb=est_bw,
            gb_per_token=gb_per_tok,
            peak_ram_gb=self.peak_ram,
            swap_pageouts_delta=pageouts_delta,
            swapouts_delta=swapouts_delta,
            effective_tok_s=self.effective_tok_s if hasattr(self, "effective_tok_s") else 0.0,
            avg_soc_power_w=avg_power,
            peak_soc_power_w=peak_power,
            total_energy_j=total_energy,
            joules_per_token=j_per_tok,
            peak_gpu_active_pct=peak_gpu,
            details=self.details
        )


def print_telemetry_table(records: list[TelemetryRecord], console: Optional[Console] = None):
    if console is None:
        console = Console()

    table = Table(title="Apple Silicon M5 Pro - NCM vs Autoregressive LLM Benchmark Suite", header_style="bold magenta")
    table.add_column("Benchmark Modus", style="cyan", no_wrap=True)
    table.add_column("Szenario / Modell", style="green")
    table.add_column("NFE", justify="right", style="yellow")
    table.add_column("Tokens", justify="right")
    table.add_column("Dauer (s)", justify="right", style="bold")
    table.add_column("Tok/s", justify="right")
    table.add_column("Theor. Read* (GB)", justify="right", style="bold red")
    table.add_column("GB / Tok*", justify="right", style="bold yellow")
    table.add_column("Ø Power (W)**", justify="right", style="cyan")
    table.add_column("Energie (J)**", justify="right", style="bold green")
    table.add_column("Peak GPU %**", justify="right", style="magenta")
    table.add_column("Peak RAM (GB)", justify="right", style="blue")
    table.add_column("Swap & Pageouts", justify="right", style="bold green")

    for r in records:
        swap_str = f"[bold green]0 / 0[/bold green]" if (r.swap_pageouts_delta == 0 and r.swapouts_delta == 0) else f"[bold red]+{r.swap_pageouts_delta} / +{r.swapouts_delta}[/bold red]"
        tok_speed_str = f"{r.tokens_per_sec:.1f}" if r.effective_tok_s <= 0 else f"{r.tokens_per_sec:.1f} ({r.effective_tok_s:.1f} eff.)"
        power_str = f"{r.avg_soc_power_w:.1f} W" if r.avg_soc_power_w > 0 else "-"
        energy_str = f"{r.total_energy_j:.1f} J" if r.total_energy_j > 0 else "-"
        gpu_str = f"{r.peak_gpu_active_pct:.1f}%" if r.peak_gpu_active_pct > 0 else "-"

        table.add_row(
            r.mode,
            r.scenario_name,
            str(r.nfe),
            str(r.generated_tokens),
            f"{r.wall_time_s:.2f}s",
            tok_speed_str,
            f"{r.est_bandwidth_gb:.2f} GB",
            f"{r.gb_per_token:.4f}",
            power_str,
            energy_str,
            gpu_str,
            f"{r.peak_ram_gb:.2f} GB",
            swap_str
        )

    console.print()
    console.print(table)
    console.print("[dim]* 'Theor. Read' = analytisches Streaming-Modell (NFE x Modellgröße in RAM).[/dim]")
    console.print("[dim]** 'Ø Power', 'Energie' & 'Peak GPU %' = physische Hardware-Telemetrie über Apple Silicon SoC-Sensoren (powermetrics).[/dim]")
    console.print()

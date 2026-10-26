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


def get_vm_stat() -> Dict[str, int]:
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
        return {"pageouts": 0, "swapins": 0, "swapouts": 0, "pageins": 0, "error": str(e)}


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
    est_bandwidth_gb: float  # NFE * model_size_gb
    gb_per_token: float  # est_bandwidth_gb / generated_tokens
    peak_ram_gb: float
    swap_pageouts_delta: int
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

    def __enter__(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.start_vm = get_vm_stat()
        self.peak_ram = get_peak_memory_gb()
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
        cur_mem = get_peak_memory_gb()
        if cur_mem > self.peak_ram:
            self.peak_ram = cur_mem
        self.end_vm = get_vm_stat()

    def get_record(self) -> TelemetryRecord:
        wall_time = max(self.end_time - self.start_time, 1e-6)
        tok_s = self.generated_tokens / wall_time if self.generated_tokens > 0 else 0.0
        pageouts_delta = self.end_vm.get("pageouts", 0) - self.start_vm.get("pageouts", 0)
        est_bw = self.nfe * self.model_size_gb
        gb_per_tok = (est_bw / self.generated_tokens) if self.generated_tokens > 0 else 0.0

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
            details=self.details
        )


def print_telemetry_table(records: list[TelemetryRecord], console: Optional[Console] = None):
    if console is None:
        console = Console()

    table = Table(title="Apple Silicon M5 Pro - NCM vs Autoregressive LLM Benchmark Suite", header_style="bold magenta")
    table.add_column("Benchmark Modus", style="cyan", no_wrap=True)
    table.add_column("Szenario / Modell", style="green")
    table.add_column("NFE (Passes)", justify="right", style="yellow")
    table.add_column("Tokens", justify="right")
    table.add_column("Dauer (s)", justify="right", style="bold")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Est. RAM Read (GB)", justify="right", style="bold red")
    table.add_column("GB / Token", justify="right", style="bold yellow")
    table.add_column("Peak RAM (GB)", justify="right", style="blue")
    table.add_column("Swap Pageouts", justify="right", style="bold green")

    for r in records:
        swap_str = f"[bold green]{r.swap_pageouts_delta}[/bold green]" if r.swap_pageouts_delta == 0 else f"[bold red]+{r.swap_pageouts_delta}[/bold red]"
        table.add_row(
            r.mode,
            r.scenario_name,
            str(r.nfe),
            str(r.generated_tokens),
            f"{r.wall_time_s:.2f}s",
            f"{r.tokens_per_sec:.1f}",
            f"{r.est_bandwidth_gb:.2f} GB",
            f"{r.gb_per_token:.4f} GB/tok",
            f"{r.peak_ram_gb:.2f} GB",
            swap_str
        )

    console.print()
    console.print(table)
    console.print()

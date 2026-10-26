#!/usr/bin/env python3
"""
Physical Hardware Profiler for Apple Silicon M5 Pro via powermetrics.
Captures real-time physical power (mW), GPU utilization (%), CPU power, and computes total SoC energy (Joules).
Uses file-based capture to avoid IPC/pipe bottlenecks.
"""

import os
import subprocess
import time
import plistlib
import tempfile
from typing import Dict, Any, List


class PhysicalHardwareProfiler:
    """
    Spawns background powermetrics sampling SoC power, GPU active residency, and clocks.
    Calculates total electrical energy consumed (Joules) and average GPU compute utilization.
    """

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self.temp_file = tempfile.NamedTemporaryFile(prefix="pm_", suffix=".plist", delete=False).name
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def start(self):
        """Starts real-time powermetrics sampling."""
        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except Exception:
                pass

        self.start_time = time.time()
        cmd = [
            "sudo", "-n", "/usr/bin/powermetrics",
            "-o", self.temp_file,
            "-f", "plist",
            "-i", str(self.sample_interval_ms),
            "-s", "cpu_power,gpu_power"
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Brief warmup to ensure powermetrics has started writing
            time.sleep(0.15)
        except Exception:
            self.proc = None

    def stop(self) -> Dict[str, Any]:
        """Stops sampling and computes physical hardware metrics from the captured trace."""
        self.end_time = time.time()

        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        duration = max(self.end_time - self.start_time, 1e-6)

        samples: List[Dict[str, Any]] = []
        if os.path.exists(self.temp_file):
            try:
                with open(self.temp_file, "rb") as f:
                    content = f.read()

                parts = content.split(b"\x00")
                for part in parts:
                    part = part.strip()
                    if not part or b"<plist" not in part:
                        continue
                    try:
                        data = plistlib.loads(part)
                        gpu = data.get("gpu", {})
                        proc = data.get("processor", {})

                        idle_ratio = gpu.get("idle_ratio", 1.0)
                        gpu_active_pct = max(0.0, min(100.0, (1.0 - idle_ratio) * 100.0))
                        gpu_freq_mhz = gpu.get("freq_hz", 0) / 1e6

                        gpu_power_mw = float(proc.get("gpu_power", 0.0))
                        cpu_power_mw = float(proc.get("cpu_power", 0.0))
                        combined_power_mw = float(proc.get("combined_power", gpu_power_mw + cpu_power_mw))

                        samples.append({
                            "gpu_active_pct": gpu_active_pct,
                            "gpu_freq_mhz": gpu_freq_mhz,
                            "gpu_power_mw": gpu_power_mw,
                            "cpu_power_mw": cpu_power_mw,
                            "combined_power_mw": combined_power_mw
                        })
                    except Exception:
                        pass
                os.remove(self.temp_file)
            except Exception:
                pass

        if not samples:
            return {
                "available": False,
                "duration_s": duration,
                "avg_gpu_active_pct": 0.0,
                "peak_gpu_active_pct": 0.0,
                "avg_soc_power_w": 0.0,
                "peak_soc_power_w": 0.0,
                "total_energy_joules": 0.0,
                "sample_count": 0
            }

        gpu_actives = [s["gpu_active_pct"] for s in samples]
        soc_powers_w = [s["combined_power_mw"] / 1000.0 for s in samples]

        avg_gpu_active = sum(gpu_actives) / len(gpu_actives)
        peak_gpu_active = max(gpu_actives)

        avg_soc_power = sum(soc_powers_w) / len(soc_powers_w)
        peak_soc_power = max(soc_powers_w)

        total_energy_j = avg_soc_power * duration

        return {
            "available": True,
            "duration_s": duration,
            "avg_gpu_active_pct": avg_gpu_active,
            "peak_gpu_active_pct": peak_gpu_active,
            "avg_soc_power_w": avg_soc_power,
            "peak_soc_power_w": peak_soc_power,
            "total_energy_joules": total_energy_j,
            "sample_count": len(samples)
        }

#!/usr/bin/env python3
"""
Table Invariant Verification Script for README.md:
Checks:
- Invariant A (Zeit & Durchsatz): Tokens / Dauer_s == Durchsatz (auf 1 Nachkommastelle)
- Invariant B (Leistung & Energie): Power_W * Dauer_s == Energy_J und (Power_W - P_idle) == Dyn_Power_W
"""

p_idle = 1.59

# Table 1: Großstichprobe N = 500
# Spalten: Modell, Passes, Ø BPE-Tokens, Dauer/Sample (ms), Iso-Latenz (28 tok) (ms), BPE Durchsatz, ROUGE-1, ROUGE-2, ROUGE-L
table1_rows = [
    {"name": "Qwen2.5-0.5B (AR Baseline)", "tokens": 28.10, "dur_s": 0.1784, "iso_dur_s": 0.1778, "throughput": 157.5},
    {"name": "Qwen2.5-0.5B (AR MLX 4-bit)", "tokens": 35.44, "dur_s": 0.1067, "iso_dur_s": 0.0843, "throughput": 332.1},
    {"name": "NCM Fast (Naive Eager Port)", "tokens": 25.94, "dur_s": 0.3800, "iso_dur_s": 0.3799, "throughput": 68.3},
    {"name": "NCM Fast (Systems-Optimized)", "tokens": 25.94, "dur_s": 0.0898, "iso_dur_s": 0.0903, "throughput": 288.9},
    {"name": "Hybrid (NCM + MLX 4-Bit)*", "tokens": 176.0, "dur_s": 3.3900, "iso_dur_s": None, "throughput": 51.9},
]

print("=== CHECKING TABLE 1 (INVARIANT A) ===")
for r in table1_rows:
    calc_tp = r["tokens"] / r["dur_s"]
    diff = abs(calc_tp - r["throughput"])
    print(f"{r['name']}: Tokens={r['tokens']}, Dur={r['dur_s']}s -> Calc TP={calc_tp:.1f}, Reported TP={r['throughput']:.1f} (Diff: {diff:.2f})")
    assert diff < 0.2, f"Table 1 invariant failed for {r['name']}: {calc_tp:.1f} vs {r['throughput']:.1f}"

# Table 3: Powermetrics (Burst-Lauf)
table3_rows = [
    {"name": "Qwen2.5-0.5B (MLX 4-Bit Metal)", "tokens": 2250, "dur_s": 6.52, "throughput": 345.1, "p_avg": 25.81, "p_dyn": 24.22, "e_total": 168.28, "j_per_tok": 0.0748, "j_dyn": 0.0702},
    {"name": "Qwen2.5-0.5B (PyTorch MPS FP16)", "tokens": 960, "dur_s": 5.70, "throughput": 168.4, "p_avg": 19.19, "p_dyn": 17.60, "e_total": 109.38, "j_per_tok": 0.1139, "j_dyn": 0.1045},
    {"name": "NCM Fast (Systems-Optimized)", "tokens": 1679, "dur_s": 5.41, "throughput": 310.4, "p_avg": 19.86, "p_dyn": 18.27, "e_total": 107.44, "j_per_tok": 0.0640, "j_dyn": 0.0589},
]

print("\n=== CHECKING TABLE 3 (INVARIANTS A & B) ===")
for r in table3_rows:
    # Invariant A: TP = Tokens / dur_s
    calc_tp = r["tokens"] / r["dur_s"]
    assert abs(calc_tp - r["throughput"]) < 0.2, f"Throughput failed for {r['name']}: {calc_tp} vs {r['throughput']}"
    # Invariant B1: E_total = P_avg * dur_s
    calc_e = r["p_avg"] * r["dur_s"]
    assert abs(calc_e - r["e_total"]) < 0.05, f"Energy failed for {r['name']}: {calc_e} vs {r['e_total']}"
    # Invariant B2: P_dyn = P_avg - p_idle
    calc_pdyn = r["p_avg"] - p_idle
    assert abs(calc_pdyn - r["p_dyn"]) < 0.05, f"Dyn Power failed for {r['name']}: {calc_pdyn} vs {r['p_dyn']}"
    # Invariant B3: J/tok = E_total / tokens
    calc_j = r["e_total"] / r["tokens"]
    assert abs(calc_j - r["j_per_tok"]) < 0.0005, f"J/tok failed for {r['name']}: {calc_j} vs {r['j_per_tok']}"
    print(f"{r['name']}: Invariants A & B PERFECTLY VALIDATED!")

# Table 4: MBR Ablation (N = 20)
table4_rows = [
    {"name": "K=1 (Fast)", "tokens": 27.98, "dur_s": 0.098, "throughput": 285.4},
    {"name": "K=4 (Low Effort)", "tokens": 27.98, "dur_s": 0.454, "throughput": 61.6},
    {"name": "K=8 (Medium Effort)", "tokens": 27.98, "dur_s": 0.897, "throughput": 31.2},
]
print("\n=== CHECKING TABLE 4 (INVARIANT A) ===")
for r in table4_rows:
    calc_tp = r["tokens"] / r["dur_s"]
    assert abs(calc_tp - r["throughput"]) < 0.2, f"Table 4 failed for {r['name']}: {calc_tp} vs {r['throughput']}"
    print(f"{r['name']}: Invariant A VALIDATED ({r['tokens']} tok / {r['dur_s']}s = {calc_tp:.1f} tok/s)!")

print("\nALL INVARIANTS A & B 100% MATHEMATICALLY VERIFIED!")

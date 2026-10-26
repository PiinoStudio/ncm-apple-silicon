#!/usr/bin/env python3
"""
MBR Candidate Scaling Ablation over N = 20 samples (passages[0:20])
Evaluates K = 1, 4, 8 candidates:
Measures latency, effective throughput, ROUGE-1, ROUGE-2, ROUGE-L, and MBR utility score.
"""

import pickle
import time
import numpy as np
from rouge_score import rouge_scorer
from src.ncm_mps_runner import NCMRunner
from src.deep_compute import DeepComputeRunner, compute_mbr_consensus

def run_mbr_ablation(n_samples: int = 20):
    print(f"=== MBR CANDIDATE SCALING ABLATION (N = {n_samples}) ===")
    passages = pickle.load(open("data/xsum/passages.pickle", "rb"))[:n_samples]
    refs = pickle.load(open("data/xsum/reference_summaries.pickle", "rb"))[:n_samples]

    runner = NCMRunner(task="xsum", device="mps")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    deep_compute = DeepComputeRunner(runner=runner)

    configs = [
        ("K=1 (Fast)", 1),
        ("K=4 (Low Effort)", 4),
        ("K=8 (Medium Effort)", 8),
    ]

    ablation_results = {}

    for name, K in configs:
        print(f"\nEvaluating {name} across {n_samples} samples...")
        t0 = time.perf_counter()
        r1_list, r2_list, rl_list = [], [], []
        mbr_utilities = []
        sample_durations = []

        for i in range(n_samples):
            p = passages[i]
            ref = refs[i]
            t_sample_start = time.perf_counter()

            if K == 1:
                res = runner.generate_coarse_to_fine(p, use_optimizations=True)
                best_text = res["text"]
                mbr_u = 1.0
            else:
                # Generate K candidates
                candidates = []
                for k in range(K):
                    temp = 0.7 + (k % 4) * 0.15
                    top_p = 0.90 + (k % 3) * 0.04 if k > 0 else 0.0
                    top_k = 50 if k > 0 else 1
                    active_cfg = 1.0 if (k % 2 == 1 and runner.net.prefix_pad_id is not None) else 0.0
                    c_res = runner.generate_coarse_to_fine(
                        p, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=active_cfg, use_optimizations=True
                    )
                    candidates.append(c_res["text"])

                best_idx, best_text, scores = compute_mbr_consensus(candidates)
                mbr_u = scores[best_idx]

            t_sample = time.perf_counter() - t_sample_start
            sample_durations.append(t_sample)
            mbr_utilities.append(mbr_u)

            s = scorer.score(ref, best_text)
            r1_list.append(s["rouge1"].fmeasure)
            r2_list.append(s["rouge2"].fmeasure)
            rl_list.append(s["rougeL"].fmeasure)

        total_time = time.perf_counter() - t0
        mean_dur = np.mean(sample_durations)
        # 48 NCM positions * 0.583 ratio = ~28 BPE tokens
        eff_bpe_tok_s = (48 * 0.583) / mean_dur

        ablation_results[name] = {
            "K": K,
            "passes": 9 * K,
            "mean_dur_s": mean_dur,
            "eff_bpe_tok_s": eff_bpe_tok_s,
            "r1": (np.mean(r1_list), np.std(r1_list)),
            "r2": (np.mean(r2_list), np.std(r2_list)),
            "rl": (np.mean(rl_list), np.std(rl_list)),
            "mbr_utility": np.mean(mbr_utilities) if K > 1 else None,
        }

        print(f"  ✓ {name}: Mean Duration = {mean_dur:.3f}s, Eff Durchsatz = {eff_bpe_tok_s:.1f} BPE tok/s")
        print(f"    ROUGE-1: {np.mean(r1_list):.4f} +/- {np.std(r1_list):.4f}")
        print(f"    ROUGE-2: {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}")
        print(f"    ROUGE-L: {np.mean(rl_list):.4f} +/- {np.std(rl_list):.4f}")
        if K > 1:
            print(f"    Ø MBR Utility Score: {np.mean(mbr_utilities):.4f}")

    runner.cleanup()

    print("\n=== SUMMARY TABLE (N = 20) ===")
    print(f"{'K':<22} | {'Passes':<6} | {'Dauer (s)':<9} | {'Eff. Durchsatz':<14} | {'ROUGE-1':<16} | {'ROUGE-2':<16} | {'ROUGE-L':<16} | {'MBR Utility'}")
    print("-" * 120)
    for name, d in ablation_results.items():
        mbr_str = f"{d['mbr_utility']:.4f}" if d['mbr_utility'] is not None else "-"
        print(f"{name:<22} | {d['passes']:<6} | {d['mean_dur_s']:<9.3f} | {d['eff_bpe_tok_s']:<14.1f} | {d['r1'][0]:.4f} +/- {d['r1'][1]:.3f} | {d['r2'][0]:.4f} +/- {d['r2'][1]:.3f} | {d['rl'][0]:.4f} +/- {d['rl'][1]:.3f} | {mbr_str}")

if __name__ == "__main__":
    run_mbr_ablation(20)

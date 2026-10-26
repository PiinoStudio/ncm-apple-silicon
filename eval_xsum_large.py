#!/usr/bin/env python3
"""
Large-Scale Statistical Evaluation on XSum (N = 500 samples)
Calculates ROUGE-1, ROUGE-2, and ROUGE-L with bootstrap significance test.
"""

import pickle
import time
import numpy as np
from rouge_score import rouge_scorer
from src.ncm_mps_runner import NCMRunner
from src.baseline_ar import AutoregressiveBaselineRunner

def run_large_eval(n_samples: int = 500):
    print(f"=== LARGE-SCALE XSUM EVALUATION (N = {n_samples}) ===")
    passages = pickle.load(open("data/xsum/passages.pickle", "rb"))[:n_samples]
    refs = pickle.load(open("data/xsum/reference_summaries.pickle", "rb"))[:n_samples]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # 1. NCM Fast Optimized
    print("\n[1/2] Evaluating NCM Fast (Systems-Optimized)...")
    ncm_runner = NCMRunner(task="xsum", device="mps")
    t0 = time.perf_counter()
    ncm_r1, ncm_r2, ncm_rl = [], [], []
    for i in range(n_samples):
        res = ncm_runner.generate_coarse_to_fine(passages[i], use_optimizations=True)
        s = scorer.score(refs[i], res["text"])
        ncm_r1.append(s["rouge1"].fmeasure)
        ncm_r2.append(s["rouge2"].fmeasure)
        ncm_rl.append(s["rougeL"].fmeasure)
        if (i + 1) % 100 == 0:
            print(f"  NCM Progress: {i+1}/{n_samples} ({time.perf_counter()-t0:.1f}s)")
    t_ncm = time.perf_counter() - t0
    ncm_runner.cleanup()

    print(f"  ✓ NCM Complete in {t_ncm:.2f}s ({(t_ncm)/n_samples*1000:.1f} ms/sample)")
    print(f"    ROUGE-1: {np.mean(ncm_r1):.4f} +/- {np.std(ncm_r1):.4f}")
    print(f"    ROUGE-2: {np.mean(ncm_r2):.4f} +/- {np.std(ncm_r2):.4f}")
    print(f"    ROUGE-L: {np.mean(ncm_rl):.4f} +/- {np.std(ncm_rl):.4f}")

    # 2. Qwen2.5-0.5B AR Baseline
    print("\n[2/2] Evaluating Qwen2.5-0.5B (AR Baseline, Zero-Shot)...")
    ar_runner = AutoregressiveBaselineRunner()
    t0 = time.perf_counter()
    ar_r1, ar_r2, ar_rl = [], [], []
    for i in range(n_samples):
        src_text = tok.decode(passages[i][:250])
        prompt = (
            f"<|im_start|>user\n"
            f"Summarize the following news article in one concise sentence:\n"
            f"{src_text}\n<|im_end|>\n<|im_start|>assistant\n"
        )
        res = ar_runner.generate(prompt, max_new_tokens=48)
        s = scorer.score(refs[i], res["text"])
        ar_r1.append(s["rouge1"].fmeasure)
        ar_r2.append(s["rouge2"].fmeasure)
        ar_rl.append(s["rougeL"].fmeasure)
        if (i + 1) % 100 == 0:
            print(f"  AR Progress: {i+1}/{n_samples} ({time.perf_counter()-t0:.1f}s)")
    t_ar = time.perf_counter() - t0
    ar_runner.cleanup()

    print(f"  ✓ AR Complete in {t_ar:.2f}s ({(t_ar)/n_samples*1000:.1f} ms/sample)")
    print(f"    ROUGE-1: {np.mean(ar_r1):.4f} +/- {np.std(ar_r1):.4f}")
    print(f"    ROUGE-2: {np.mean(ar_r2):.4f} +/- {np.std(ar_r2):.4f}")
    print(f"    ROUGE-L: {np.mean(ar_rl):.4f} +/- {np.std(ar_rl):.4f}")

    # Paired Bootstrap Test (10,000 resamples)
    print("\n=== PAIRED BOOTSTRAP SIGNIFICANCE TEST (N = 500, 10,000 RESAMPLES) ===")
    np.random.seed(42)
    diff_r1 = np.array(ncm_r1) - np.array(ar_r1)
    diff_r2 = np.array(ncm_r2) - np.array(ar_r2)
    diff_rl = np.array(ncm_rl) - np.array(ar_rl)

    def bootstrap_p(diff):
        boot_diffs = np.random.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        p = np.mean(boot_diffs <= 0)
        return float(p)

    p1 = bootstrap_p(diff_r1)
    p2 = bootstrap_p(diff_r2)
    pl = bootstrap_p(diff_rl)

    print(f"ROUGE-1: Delta = +{np.mean(diff_r1):.4f}, p = {p1:.4f}")
    print(f"ROUGE-2: Delta = +{np.mean(diff_r2):.4f}, p = {p2:.4f}")
    print(f"ROUGE-L: Delta = +{np.mean(diff_rl):.4f}, p = {pl:.4f}")

if __name__ == "__main__":
    run_large_eval(500)

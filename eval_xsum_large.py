#!/usr/bin/env python3
"""
Large-Scale Statistical Evaluation on XSum (N = 500 samples)
Evaluates:
  1. NCM Fast (Systems-Optimized)
  2. Qwen2.5-0.5B (AR PyTorch MPS FP16, Zero-Shot)
  3. Qwen2.5-0.5B (AR Apple MLX 4-Bit Metal, Zero-Shot)
Calculates exact generated token counts, latency, throughput, and ROUGE-1/2/L with paired bootstrap test.
"""

import pickle
import time
import numpy as np
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from src.ncm_mps_runner import NCMRunner
from src.baseline_ar import AutoregressiveBaselineRunner

def run_large_eval(n_samples: int = 500):
    print(f"=== LARGE-SCALE XSUM EVALUATION (N = {n_samples}) ===")
    passages = pickle.load(open("data/xsum/passages.pickle", "rb"))[:n_samples]
    refs = pickle.load(open("data/xsum/reference_summaries.pickle", "rb"))[:n_samples]
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # 1. NCM Fast Systems-Optimized
    print("\n[1/3] Evaluating NCM Fast (Systems-Optimized)...")
    ncm_runner = NCMRunner(task="xsum", device="mps")
    t0 = time.perf_counter()
    ncm_r1, ncm_r2, ncm_rl = [], [], []
    ncm_bpe_toks = []
    for i in range(n_samples):
        res = ncm_runner.generate_coarse_to_fine(passages[i], use_optimizations=True)
        s = scorer.score(refs[i], res["text"])
        ncm_r1.append(s["rouge1"].fmeasure)
        ncm_r2.append(s["rouge2"].fmeasure)
        ncm_rl.append(s["rougeL"].fmeasure)
        ncm_bpe_toks.append(len(gpt2_tok.encode(res["text"])))
        if (i + 1) % 100 == 0:
            print(f"  NCM Progress: {i+1}/{n_samples} ({time.perf_counter()-t0:.1f}s)")
    t_ncm = time.perf_counter() - t0
    ncm_runner.cleanup()

    print(f"  ✓ NCM Complete in {t_ncm:.2f}s ({(t_ncm)/n_samples*1000:.1f} ms/sample)")
    print(f"    Ø BPE-Tokens: {np.mean(ncm_bpe_toks):.2f} +/- {np.std(ncm_bpe_toks):.2f}")
    print(f"    BPE Throughput: {sum(ncm_bpe_toks)/t_ncm:.1f} tok/s")
    print(f"    ROUGE-1: {np.mean(ncm_r1):.4f} +/- {np.std(ncm_r1):.4f}")
    print(f"    ROUGE-2: {np.mean(ncm_r2):.4f} +/- {np.std(ncm_r2):.4f}")
    print(f"    ROUGE-L: {np.mean(ncm_rl):.4f} +/- {np.std(ncm_rl):.4f}")

    # 2. Qwen2.5-0.5B AR PyTorch MPS FP16
    print("\n[2/3] Evaluating Qwen2.5-0.5B (AR PyTorch MPS FP16)...")
    ar_runner = AutoregressiveBaselineRunner()
    t0 = time.perf_counter()
    ar_r1, ar_r2, ar_rl = [], [], []
    ar_toks = []
    for i in range(n_samples):
        src_text = gpt2_tok.decode(passages[i][:250])
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
        ar_toks.append(res["generated_tokens"])
        if (i + 1) % 100 == 0:
            print(f"  AR MPS Progress: {i+1}/{n_samples} ({time.perf_counter()-t0:.1f}s)")
    t_ar = time.perf_counter() - t0
    ar_runner.cleanup()

    print(f"  ✓ AR MPS Complete in {t_ar:.2f}s ({(t_ar)/n_samples*1000:.1f} ms/sample)")
    print(f"    Ø Tokens: {np.mean(ar_toks):.2f} +/- {np.std(ar_toks):.2f}")
    print(f"    Throughput: {sum(ar_toks)/t_ar:.1f} tok/s")
    print(f"    ROUGE-1: {np.mean(ar_r1):.4f} +/- {np.std(ar_r1):.4f}")
    print(f"    ROUGE-2: {np.mean(ar_r2):.4f} +/- {np.std(ar_r2):.4f}")
    print(f"    ROUGE-L: {np.mean(ar_rl):.4f} +/- {np.std(ar_rl):.4f}")

    # 3. Qwen2.5-0.5B AR Apple MLX 4-Bit Metal
    print("\n[3/3] Evaluating Qwen2.5-0.5B (AR Apple MLX 4-Bit Metal)...")
    from mlx_lm import load, generate
    mlx_model, mlx_tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    # Warmup
    _ = generate(mlx_model, mlx_tok, prompt="Hello", max_tokens=10, verbose=False)

    t0 = time.perf_counter()
    mlx_r1, mlx_r2, mlx_rl = [], [], []
    mlx_toks = []
    for i in range(n_samples):
        src_text = gpt2_tok.decode(passages[i][:250])
        prompt = (
            f"<|im_start|>user\n"
            f"Summarize the following news article in one concise sentence:\n"
            f"{src_text}\n<|im_end|>\n<|im_start|>assistant\n"
        )
        out = generate(mlx_model, mlx_tok, prompt=prompt, max_tokens=48, verbose=False)
        s = scorer.score(refs[i], out)
        mlx_r1.append(s["rouge1"].fmeasure)
        mlx_r2.append(s["rouge2"].fmeasure)
        mlx_rl.append(s["rougeL"].fmeasure)
        mlx_toks.append(len(mlx_tok.encode(out)))
        if (i + 1) % 100 == 0:
            print(f"  MLX Progress: {i+1}/{n_samples} ({time.perf_counter()-t0:.1f}s)")
    t_mlx = time.perf_counter() - t0
    del mlx_model
    del mlx_tok

    print(f"  ✓ MLX Metal Complete in {t_mlx:.2f}s ({(t_mlx)/n_samples*1000:.1f} ms/sample)")
    print(f"    Ø Tokens: {np.mean(mlx_toks):.2f} +/- {np.std(mlx_toks):.2f}")
    print(f"    Throughput: {sum(mlx_toks)/t_mlx:.1f} tok/s")
    print(f"    ROUGE-1: {np.mean(mlx_r1):.4f} +/- {np.std(mlx_r1):.4f}")
    print(f"    ROUGE-2: {np.mean(mlx_r2):.4f} +/- {np.std(mlx_r2):.4f}")
    print(f"    ROUGE-L: {np.mean(mlx_rl):.4f} +/- {np.std(mlx_rl):.4f}")

    # Paired Bootstrap Test (10,000 resamples)
    print("\n=== PAIRED BOOTSTRAP SIGNIFICANCE TEST (N = 500, 10,000 RESAMPLES) ===")
    np.random.seed(42)
    def bootstrap_p(diff):
        boot_diffs = np.random.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        return float(np.mean(boot_diffs <= 0))

    diff_r1 = np.array(ncm_r1) - np.array(ar_r1)
    diff_r2 = np.array(ncm_r2) - np.array(ar_r2)
    diff_rl = np.array(ncm_rl) - np.array(ar_rl)
    print(f"NCM vs AR MPS FP16: ROUGE-1 Delta = +{np.mean(diff_r1):.4f} (p = {bootstrap_p(diff_r1):.4f}) | ROUGE-L Delta = +{np.mean(diff_rl):.4f} (p = {bootstrap_p(diff_rl):.4f})")

    diff_mlx_r1 = np.array(ncm_r1) - np.array(mlx_r1)
    diff_mlx_rl = np.array(ncm_rl) - np.array(mlx_rl)
    print(f"NCM vs MLX 4-Bit:   ROUGE-1 Delta = +{np.mean(diff_mlx_r1):.4f} (p = {bootstrap_p(diff_mlx_r1):.4f}) | ROUGE-L Delta = +{np.mean(diff_mlx_rl):.4f} (p = {bootstrap_p(diff_mlx_rl):.4f})")

if __name__ == "__main__":
    run_large_eval(500)

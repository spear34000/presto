"""Baseline throughput probe: HuggingFace Transformers greedy decode on CPU.

Measures the same workload presto bench measures (greedy, fixed steps) using
the world's most common inference stack, so the comparison table reflects a
real alternative rather than a strawman.
"""
import argparse
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids

    # warmup
    with torch.no_grad():
        model.generate(ids, max_new_tokens=min(args.steps, 8), do_sample=False)

    rates = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.steps, do_sample=False,
                                 pad_token_id=tok.eos_token_id or 0)
        dt = time.perf_counter() - t0
        new_tokens = out.shape[1] - ids.shape[1]
        rates.append(new_tokens / dt)

    med = statistics.median(rates)
    print(f"[hf-bench] model={args.model} steps={args.steps} "
          f"med_tps={med:.2f} min={min(rates):.2f} max={max(rates):.2f} ok=true")


if __name__ == "__main__":
    main()

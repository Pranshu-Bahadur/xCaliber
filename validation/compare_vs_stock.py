"""Compare xR64ABS vs stock vLLM on a plain-GPTQ Qwen3.6 MoE checkpoint.

Runs the same greedy generation twice, each in its OWN subprocess so the two
vLLM engines never share a process:

    * stock  : VLLM_PLUGINS=''                (marlin routed experts)
    * xr64abs: VLLM_PLUGINS='xcaliber_xr64_abs' (monolithic routed experts)

then diffs the decoded outputs (expected: similar / coherent in both) and
reports a small throughput probe for each. Both transcripts are printed
verbatim so a human can eyeball coherence.

GPU is required. NOT run during code review -- run later on an idle device:

    CUDA_VISIBLE_DEVICES=0 python validation/compare_vs_stock.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


MODEL_ID = os.environ.get("MODEL_ID", "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
GPU_MEM_UTIL = os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", "128"))

PROMPTS = (
    "Explain, in two sentences, what a mixture-of-experts shared expert does.",
    "Write a haiku about coalesced GPU memory loads.",
    "What is 248 * 17? Show the arithmetic briefly.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
)


# --------------------------------------------------------------------------
# Worker: runs inside a fresh subprocess, emits one JSON blob on stdout.
# --------------------------------------------------------------------------
def run_worker() -> int:
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=int(MAX_MODEL_LEN),
        gpu_memory_utilization=float(GPU_MEM_UTIL),
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=0)
    outputs = llm.generate(prompts, greedy, use_tqdm=False)
    texts = [o.outputs[0].text.strip() for o in outputs]

    # Throughput probe: longer greedy decode over the same prompts.
    probe = SamplingParams(temperature=0.0, max_tokens=PROBE_TOKENS, seed=0)
    start = time.perf_counter()
    probe_out = llm.generate(prompts, probe, use_tqdm=False)
    elapsed = time.perf_counter() - start
    probe_tokens = sum(len(o.outputs[0].token_ids) for o in probe_out)

    print(
        "XR64_RESULT_JSON:"
        + json.dumps(
            {
                "texts": texts,
                "probe_tokens": probe_tokens,
                "probe_seconds": elapsed,
                "probe_tok_s": probe_tokens / elapsed if elapsed else 0.0,
                "plugins": os.environ.get("VLLM_PLUGINS", ""),
            }
        ),
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def launch(label: str, plugins: str) -> dict:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = plugins
    env["HF_HOME"] = env.get("HF_HOME", "/tmp/hf-cache")
    print(f"\n=== launching [{label}] VLLM_PLUGINS={plugins!r} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker"],
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(proc.stderr)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("XR64_RESULT_JSON:"):
            result = json.loads(line[len("XR64_RESULT_JSON:"):])
    if result is None:
        print(proc.stdout)
        raise RuntimeError(f"[{label}] worker produced no result (rc={proc.returncode})")
    return result


def main() -> int:
    stock = launch("stock", "")
    xr64 = launch("xr64abs", "xcaliber_xr64_abs")

    print("\n" + "=" * 72)
    print("VERBATIM OUTPUTS")
    print("=" * 72)
    for i, prompt in enumerate(PROMPTS):
        print(f"\n----- PROMPT {i}: {prompt}")
        print(f"[stock  ] {stock['texts'][i]}")
        print(f"[xr64abs] {xr64['texts'][i]}")
        print(f"identical: {stock['texts'][i] == xr64['texts'][i]}")

    print("\n" + "=" * 72)
    print("THROUGHPUT PROBE")
    print("=" * 72)
    for label, r in (("stock", stock), ("xr64abs", xr64)):
        print(
            f"[{label:>8}] {r['probe_tokens']:5d} tok in "
            f"{r['probe_seconds']:7.3f}s -> {r['probe_tok_s']:8.2f} tok/s"
        )
    if stock["probe_tok_s"]:
        speedup = xr64["probe_tok_s"] / stock["probe_tok_s"]
        print(f"xr64abs / stock throughput ratio: {speedup:.3f}x")

    n_identical = sum(
        s == x for s, x in zip(stock["texts"], xr64["texts"])
    )
    print(f"\nidentical prompts: {n_identical}/{len(PROMPTS)}")
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(run_worker())
    raise SystemExit(main())

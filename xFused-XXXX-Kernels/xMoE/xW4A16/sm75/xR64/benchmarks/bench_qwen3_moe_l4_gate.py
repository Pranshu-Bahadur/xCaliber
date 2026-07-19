import csv
import contextlib
import os
import statistics
import sys
import time
from pathlib import Path


if "__file__" in globals():
    plugin_root = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != plugin_root
    ]

os.environ.setdefault("VLLM_PLUGINS", "xcaliber_xr64_abs")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm.distributed.parallel_state
import vllm.utils.system_utils


vllm.utils.system_utils.suppress_stdout = contextlib.nullcontext
vllm.distributed.parallel_state.suppress_stdout = contextlib.nullcontext

from vllm import LLM, SamplingParams


MODEL_ID = os.environ.get(
    "MODEL_ID",
    "RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16",
)
PROMPTS = int(os.environ.get("PROMPTS", "8"))
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "64"))
TRIALS = int(os.environ.get("TRIALS", "2"))
CSV_PATH = Path(
    os.environ.get("CSV_PATH", "/content/xr64abs_qwen3_l4_gate.csv")
)
OUTPUT_PATH = Path(
    os.environ.get("OUTPUT_PATH", "/content/xr64abs_qwen3_l4_gate.txt")
)


QUESTIONS = (
    "Explain why coalesced GPU loads improve memory throughput.",
    "Give a two-line Python function that computes Fibonacci numbers.",
    "What is the difference between latency and throughput?",
    "Summarize the purpose of a mixture-of-experts router.",
    "Name one reason quantization can accelerate inference.",
    "Solve 17 * 23 and show the arithmetic briefly.",
    "Explain CUDA shared-memory bank conflicts in one paragraph.",
    "Write one sentence describing speculative decoding.",
)


def make_prompts(tokenizer, trial):
    return [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": f"Trial {trial}. {QUESTIONS[i % len(QUESTIONS)]}",
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(PROMPTS)
    ]


gpu = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if capability != (8, 9):
    raise RuntimeError(f"xR64ABS L4 gate requires sm_89, got {capability}: {gpu}")

torch.cuda.reset_peak_memory_stats()
load_start = time.perf_counter()
llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=512,
    max_num_seqs=PROMPTS,
    max_num_batched_tokens=4096,
    gpu_memory_utilization=0.92,
    enforce_eager=True,
    enable_prefix_caching=False,
)
load_s = time.perf_counter() - load_start
load_peak_gib = torch.cuda.max_memory_reserved() / 2**30
tokenizer = llm.get_tokenizer()

warmup = SamplingParams(temperature=0.0, max_tokens=8)
llm.generate(make_prompts(tokenizer, -1), warmup, use_tqdm=False)

sampling = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=20,
    max_tokens=OUTPUT_TOKENS,
    seed=0,
)
rows = []
captured_outputs = None
for trial in range(TRIALS):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(
        make_prompts(tokenizer, trial),
        sampling,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    generation_s = time.perf_counter() - start
    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    if output_tokens <= 0 or any(not o.outputs[0].text for o in outputs):
        raise RuntimeError("xR64ABS L4 generation produced an empty output")
    rows.append({
        "kind": "trial",
        "trial": trial,
        "gpu": gpu,
        "compute_capability": "8.9",
        "model": MODEL_ID,
        "prompts": PROMPTS,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "model_load_s": load_s,
        "load_peak_gib": load_peak_gib,
        "generation_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
        "generation_s": generation_s,
        "aggregate_output_tok_s": output_tokens / generation_s,
    })
    if captured_outputs is None:
        captured_outputs = outputs

rows.append({
    "kind": "median",
    "trial": "",
    "gpu": gpu,
    "compute_capability": "8.9",
    "model": MODEL_ID,
    "prompts": PROMPTS,
    "prompt_tokens": statistics.median(r["prompt_tokens"] for r in rows),
    "output_tokens": statistics.median(r["output_tokens"] for r in rows),
    "model_load_s": load_s,
    "load_peak_gib": load_peak_gib,
    "generation_peak_gib": statistics.median(
        r["generation_peak_gib"] for r in rows
    ),
    "generation_s": statistics.median(r["generation_s"] for r in rows),
    "aggregate_output_tok_s": statistics.median(
        r["aggregate_output_tok_s"] for r in rows
    ),
})

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with CSV_PATH.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

blocks = []
for index, output in enumerate(captured_outputs):
    response = output.outputs[0]
    blocks.extend((
        f"===== OUTPUT {index} =====",
        f"PROMPT_TOKENS: {len(output.prompt_token_ids)}",
        f"OUTPUT_TOKENS: {len(response.token_ids)}",
        f"FINISH_REASON: {response.finish_reason}",
        response.text,
        "",
    ))
OUTPUT_PATH.write_text("\n".join(blocks))

print(
    f"gpu={gpu} sm=89 load_s={load_s:.3f} "
    f"load_peak_gib={load_peak_gib:.3f}"
)
print("+--------+--------------+---------------+------------------+----------+")
print("| trial  | generation_s | output_tokens | aggregate tok/s  | peak GiB |")
print("+--------+--------------+---------------+------------------+----------+")
for row in rows:
    label = "median" if row["kind"] == "median" else str(row["trial"])
    print(
        f"| {label:>6} | {row['generation_s']:12.3f} |"
        f" {row['output_tokens']:13.0f} |"
        f" {row['aggregate_output_tok_s']:16.2f} |"
        f" {row['generation_peak_gib']:8.3f} |"
    )
print("+--------+--------------+---------------+------------------+----------+")
print("correctness=PASS (all prompts returned non-empty outputs)")
print(f"wrote {CSV_PATH}")
print(f"wrote {OUTPUT_PATH}")

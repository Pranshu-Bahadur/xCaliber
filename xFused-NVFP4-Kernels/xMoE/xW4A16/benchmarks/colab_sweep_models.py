import csv
import subprocess
from pathlib import Path


root = Path("/content/xcalibrr_contiguous_direct")
build = root / "build"
build.mkdir(parents=True, exist_ok=True)
exe = build / "benchmark"

models = {
    "kimi_k2": (384, 2048, 7168, 8),
    "qwen3_30b_a3b": (128, 768, 2048, 8),
    "qwen3_235b_a22b": (128, 1536, 4096, 8),
    "minimax_m2": (256, 1536, 3072, 8),
    "gpt_oss_120b": (128, 2880, 2880, 4),
    "gpt_oss_20b": (32, 2880, 2880, 4),
    "gemma4_26b_a4b": (128, 704, 2816, 8),
}
Ns = (8, 256, 512)
routes = ("uniform", "zipf", "burst")
base_seed = 0x6A09E667
iters = 20

result = subprocess.run(
    [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-arch=sm_120a",
        "-Xptxas=-v",
        str(root / "benchmark.cu"),
        "-o",
        str(exe),
    ],
    text=True,
    capture_output=True,
)
log = result.stdout + result.stderr
print(log, end="", flush=True)
(root / "ptxas_full_sweep.log").write_text(log)
result.check_returncode()

rows = []
case = 0
for model, (E, I, H, TOPK) in models.items():
    for N in Ns:
        for routing in routes:
            seed = (base_seed ^ (case * 0x9E3779B9)) & 0xFFFFFFFF
            output = root / "sweep_parts" / f"{case:03d}_{model}_{N}_{routing}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            run = subprocess.run(
                [
                    str(exe), str(E), str(N), str(I), str(H), str(TOPK),
                    routing, hex(seed), str(iters), str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if run.returncode:
                print(run.stdout, end="", flush=True)
                run.check_returncode()
            with output.open(newline="") as f:
                row = next(csv.DictReader(f))
            row = {"model": model, **row}
            rows.append(row)
            print(
                f"[{case + 1:02d}/63] {model:20s} N={N:3d} {routing:7s} "
                f"{float(row['end_to_end_ms']):9.6f} ms {row['status']}",
                flush=True,
            )
            case += 1

combined = root / "results_models.csv"
with combined.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary_rows = []
for model, (E, I, H, TOPK) in models.items():
    for N in Ns:
        group = [row for row in rows if row["model"] == model and int(row["N"]) == N]
        by_route = {row["routing"]: row for row in group}
        out = {
            "model": model,
            "E": E,
            "N": N,
            "I": I,
            "H": H,
            "TOPK": TOPK,
        }
        for routing in routes:
            row = by_route[routing]
            for name in (
                "live_experts",
                "max_tokens_per_expert",
                "active_n16",
                "paired_n16",
                "end_to_end_ms",
                "useful_tflops",
                "status",
            ):
                out[f"{routing}_{name}"] = row[name]
        out["mean_end_to_end_ms"] = f"{sum(float(row['end_to_end_ms']) for row in group) / 3.0:.6f}"
        out["status"] = "PASS" if all(row["status"] == "PASS" for row in group) else "FAIL"
        summary_rows.append(out)

summary = root / "results_models_summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(combined, flush=True)
print(summary, flush=True)

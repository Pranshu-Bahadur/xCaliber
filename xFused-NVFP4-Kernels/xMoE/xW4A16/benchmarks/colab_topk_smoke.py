import csv
import subprocess
from pathlib import Path


root = Path("/content/xcalibrr_topk")
build = root / "build"
build.mkdir(parents=True, exist_ok=True)
exe = build / "benchmark"

result = subprocess.run(
    [
        "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_120a",
        "-Xptxas=-v", str(root / "benchmark.cu"), "-o", str(exe),
    ],
    text=True,
    capture_output=True,
)
log = result.stdout + result.stderr
print(log, end="", flush=True)
(root / "ptxas_topk.log").write_text(log)
result.check_returncode()

cases = (
    (128, 8, 768, 2048, 8, "uniform", "softmax"),
    (128, 8, 768, 2048, 8, "uniform", "sigmoid"),
    (384, 256, 2048, 7168, 8, "zipf", "softmax"),
    (384, 256, 2048, 7168, 8, "zipf", "sigmoid"),
)
rows = []
for case, (E, N, I, H, TOPK, routing, gate) in enumerate(cases):
    output = root / f"topk_{case}_{gate}.csv"
    run = subprocess.run(
        [
            str(exe), str(E), str(N), str(I), str(H), str(TOPK),
            routing, hex(0x6A09E667 + case), "5", str(output), gate,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(run.stdout, end="", flush=True)
    run.check_returncode()
    with output.open(newline="") as f:
        rows.append(next(csv.DictReader(f)))

combined = root / "results_topk_smoke.csv"
with combined.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(combined, flush=True)

import csv
import os
import subprocess
from pathlib import Path

import numpy as np


root = Path("/content/xcalibrr_contiguous_direct")
baseline_root = root / "baseline"
baseline_build = baseline_root / "build"
baseline_build.mkdir(parents=True, exist_ok=True)
baseline = baseline_build / "benchmark"
direct = root / "build" / "benchmark"

result = subprocess.run(
    [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-arch=sm_120a",
        "-Xptxas=-v",
        str(baseline_root / "benchmark.cu"),
        "-o",
        str(baseline),
    ],
    text=True,
    capture_output=True,
)
log = result.stdout + result.stderr
print(log, end="", flush=True)
(root / "ptxas_baseline.log").write_text(log)
result.check_returncode()

profiles = tuple(
    zip(
        ("uniform", "zipf", "burst"),
        (0x6A09E667, 0xF43E9FDE, 0x56671515),
    )
)
baseline_rows = []
for routing, seed in profiles:
    output = root / f"results_baseline_samevm_{routing}.csv"
    subprocess.run(
        [
            str(baseline), "384", "512", "2048", "7168", "8",
            routing, hex(seed), "20", str(output),
        ],
        check=True,
    )
    with output.open(newline="") as f:
        baseline_rows.extend(csv.DictReader(f))

output = root / "results_baseline_samevm.csv"
with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=baseline_rows[0])
    writer.writeheader()
    writer.writerows(baseline_rows)

env = dict(os.environ, XCALIBER_DUMP_OUTPUT="1")
compare_rows = []
for routing, seed in profiles:
    outputs = {}
    metrics = {}
    for name, exe in (("baseline", baseline), ("direct", direct)):
        result_path = root / f"numeric_{name}_{routing}.csv"
        subprocess.run(
            [
                str(exe), "384", "512", "2048", "7168", "8",
                routing, hex(seed), "1", str(result_path),
            ],
            check=True,
            env=env,
        )
        with result_path.open(newline="") as f:
            metrics[name] = next(csv.DictReader(f))
        outputs[name] = np.fromfile(f"{result_path}.out.bf16", dtype=np.uint16)

    base = (outputs["baseline"].astype(np.uint32) << 16).view(np.float32)
    test = (outputs["direct"].astype(np.uint32) << 16).view(np.float32)
    delta = np.abs(base - test)
    scale = np.maximum(np.abs(base), np.float32(1.0e-12))
    compare_rows.append(
        {
            "routing": routing,
            "bf16_exact_percent": f"{100.0 * np.mean(outputs['baseline'] == outputs['direct']):.6f}",
            "mean_abs_delta": f"{float(np.mean(delta)):.9g}",
            "max_abs_delta": f"{float(np.max(delta)):.9g}",
            "max_rel_delta": f"{float(np.max(delta / scale)):.9g}",
            "baseline_status": metrics["baseline"]["status"],
            "direct_status": metrics["direct"]["status"],
        }
    )

output = root / "results_output_delta_vs_previous_w4a4.csv"
with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=compare_rows[0])
    writer.writeheader()
    writer.writerows(compare_rows)
print(output)

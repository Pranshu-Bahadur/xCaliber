import csv
import os
import shutil
import subprocess
from pathlib import Path


root = Path("/content/xcalibrr_contiguous_direct")
build = root / "build"
parts = root / "profile_parts"
build.mkdir(parents=True, exist_ok=True)
parts.mkdir(parents=True, exist_ok=True)
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

metrics = (
    "gpu__time_duration.sum",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor.sum",
)


def number(value):
    return float(value.replace(",", ""))


def short_kernel(name):
    name = name.split("(", 1)[0]
    return {
        "moe_act_absmax_partial": "act_absmax",
        "moe_act_scale_finalize": "act_scale",
        "moe_route_pack_contiguous": "route_pack",
        "moe_act_pack_expert_contiguous": "act_scatter",
        "xR57F1_contiguous_graph": "ff1",
        "xR57F2_direct_reduce_graph": "ff2",
    }.get(name, name)


def parse_ncu(text):
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines)
        if line.startswith('"ID","Process ID"')
    )
    return [
        row for row in csv.DictReader(lines[start:])
        if row.get("ID", "").isdigit()
    ]


def weighted(rows, key):
    total = sum(row["kernel_ns"] for row in rows)
    return sum(row["kernel_ns"] * row[key] for row in rows) / total


compile_run = subprocess.run(
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
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(compile_run.stdout, end="", flush=True)
(root / "ptxas_gpu_util.log").write_text(compile_run.stdout)
compile_run.check_returncode()

ncu = shutil.which("ncu")
if not ncu:
    raise RuntimeError("ncu not found")

case_rows = []
kernel_rows = []
case = 0
for model, (E, I, H, TOPK) in models.items():
    for N in Ns:
        for routing in routes:
            seed = (base_seed ^ (case * 0x9E3779B9)) & 0xFFFFFFFF
            bench_csv = parts / f"{case:03d}_{model}_{N}_{routing}_bench.csv"
            ncu_log = parts / f"{case:03d}_{model}_{N}_{routing}_ncu.csv.txt"
            argv = [
                str(exe), str(E), str(N), str(I), str(H), str(TOPK),
                routing, hex(seed), str(iters), str(bench_csv),
            ]
            env = os.environ.copy()
            env["XCALIBER_PROFILE_ONCE"] = "1"
            run = subprocess.run(
                [
                    ncu,
                    "--profile-from-start", "off",
                    "--target-processes", "all",
                    "--cache-control", "none",
                    "--apply-rules", "no",
                    "--page", "raw",
                    "--csv",
                    "--metrics", ",".join(metrics),
                    *argv,
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            ncu_log.write_text(run.stdout)
            if run.returncode:
                print(run.stdout, end="", flush=True)
                run.check_returncode()

            with bench_csv.open(newline="") as f:
                bench = next(csv.DictReader(f))
            raw = parse_ncu(run.stdout)
            parsed = []
            for row in raw:
                parsed.append({
                    "kernel": short_kernel(row["Kernel Name"]),
                    "kernel_ns": number(row["gpu__time_duration.sum"]),
                    "gpu_util_pct": number(row[
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
                    ]),
                    "sm_throughput_pct": number(row[
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
                    ]),
                    "sm_active_pct": number(row[
                        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
                    ]),
                    "dram_throughput_pct": number(row[
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed"
                    ]),
                    "l2_throughput_pct": number(row[
                        "lts__throughput.avg.pct_of_peak_sustained_elapsed"
                    ]),
                    "tensor_pipe_active_pct": number(row[
                        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
                    ]),
                    "achieved_occupancy_pct": number(row[
                        "sm__warps_active.avg.pct_of_peak_sustained_active"
                    ]),
                    "tensor_inst": int(number(row[
                        "sm__inst_executed_pipe_tensor.sum"
                    ])),
                    "grid_size": int(number(row["launch__grid_size"])),
                    "block_size": int(number(row["launch__block_size"])),
                    "registers_per_thread": int(number(
                        row["launch__registers_per_thread"]
                    )),
                    "dynamic_smem_bytes": int(number(
                        row["launch__shared_mem_per_block_dynamic"]
                    )),
                })

            expected = {
                "act_absmax", "act_scale", "route_pack",
                "act_scatter", "ff1", "ff2",
            }
            got = {row["kernel"] for row in parsed}
            if got != expected:
                raise RuntimeError(
                    f"{model} N={N} {routing}: kernels {sorted(got)}"
                )

            profile_ns = sum(row["kernel_ns"] for row in parsed)
            for row in parsed:
                kernel_rows.append({
                    "model": model,
                    "routing": routing,
                    "seed": f"0x{seed:08x}",
                    "E": E,
                    "N": N,
                    "NP": (N + 31) & ~31,
                    "I": I,
                    "H": H,
                    "TOPK": TOPK,
                    "kernel": row["kernel"],
                    "kernel_ms": f"{row['kernel_ns'] / 1.0e6:.6f}",
                    "kernel_time_share_pct": f"{100.0 * row['kernel_ns'] / profile_ns:.4f}",
                    "gpu_util_pct": f"{row['gpu_util_pct']:.4f}",
                    "sm_throughput_pct": f"{row['sm_throughput_pct']:.4f}",
                    "sm_active_pct": f"{row['sm_active_pct']:.4f}",
                    "dram_throughput_pct": f"{row['dram_throughput_pct']:.4f}",
                    "l2_throughput_pct": f"{row['l2_throughput_pct']:.4f}",
                    "tensor_pipe_active_pct": f"{row['tensor_pipe_active_pct']:.4f}",
                    "achieved_occupancy_pct": f"{row['achieved_occupancy_pct']:.4f}",
                    "tensor_inst": row["tensor_inst"],
                    "grid_size": row["grid_size"],
                    "block_size": row["block_size"],
                    "registers_per_thread": row["registers_per_thread"],
                    "dynamic_smem_bytes": row["dynamic_smem_bytes"],
                })

            by_kernel = {row["kernel"]: row for row in parsed}
            ff1 = by_kernel["ff1"]
            ff2 = by_kernel["ff2"]
            overall_gpu = weighted(parsed, "gpu_util_pct")
            overall_sm = weighted(parsed, "sm_throughput_pct")
            overall_sm_active = weighted(parsed, "sm_active_pct")
            overall_dram = weighted(parsed, "dram_throughput_pct")
            overall_l2 = weighted(parsed, "l2_throughput_pct")
            overall_tensor = weighted(parsed, "tensor_pipe_active_pct")
            overall_occ = weighted(parsed, "achieved_occupancy_pct")
            bottleneck = "dram" if overall_dram >= overall_sm else "sm"
            case_rows.append({
                "model": model,
                "gpu": bench["gpu"],
                "sms": bench["sms"],
                "routing": routing,
                "seed": f"0x{seed:08x}",
                "E": E,
                "N": N,
                "NP": bench["NP"],
                "I": I,
                "H": H,
                "TOPK": TOPK,
                "live_experts": bench["live_experts"],
                "assignments": bench["assignments"],
                "max_tokens_per_expert": bench["max_tokens_per_expert"],
                "end_to_end_ms": bench["end_to_end_ms"],
                "useful_tflops": bench["useful_tflops"],
                "profiled_kernel_ms": f"{profile_ns / 1.0e6:.6f}",
                "overall_gpu_util_pct": f"{overall_gpu:.4f}",
                "sm_throughput_pct": f"{overall_sm:.4f}",
                "sm_active_pct": f"{overall_sm_active:.4f}",
                "dram_throughput_pct": f"{overall_dram:.4f}",
                "l2_throughput_pct": f"{overall_l2:.4f}",
                "tensor_pipe_active_pct": f"{overall_tensor:.4f}",
                "achieved_occupancy_pct": f"{overall_occ:.4f}",
                "ff1_time_share_pct": f"{100.0 * ff1['kernel_ns'] / profile_ns:.4f}",
                "ff1_gpu_util_pct": f"{ff1['gpu_util_pct']:.4f}",
                "ff1_dram_throughput_pct": f"{ff1['dram_throughput_pct']:.4f}",
                "ff1_tensor_pipe_active_pct": f"{ff1['tensor_pipe_active_pct']:.4f}",
                "ff2_time_share_pct": f"{100.0 * ff2['kernel_ns'] / profile_ns:.4f}",
                "ff2_gpu_util_pct": f"{ff2['gpu_util_pct']:.4f}",
                "ff2_dram_throughput_pct": f"{ff2['dram_throughput_pct']:.4f}",
                "ff2_tensor_pipe_active_pct": f"{ff2['tensor_pipe_active_pct']:.4f}",
                "bottleneck": bottleneck,
                "metric_definition": "duration_weighted_ncu_compute_memory_sol",
                "profile_scope": "six_graph_compute_kernels_async_memset_excluded",
                "status": bench["status"],
            })

            print(
                f"[{case + 1:02d}/63] {model:20s} N={N:3d} {routing:7s} "
                f"GPU={overall_gpu:6.2f}% DRAM={overall_dram:6.2f}% "
                f"tensor={overall_tensor:6.2f}% {bench['status']}",
                flush=True,
            )
            case += 1

combined = root / "results_gpu_util.csv"
with combined.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=case_rows[0].keys())
    writer.writeheader()
    writer.writerows(case_rows)

kernels = root / "results_gpu_util_kernels.csv"
with kernels.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=kernel_rows[0].keys())
    writer.writeheader()
    writer.writerows(kernel_rows)

summary_rows = []
for model, (E, I, H, TOPK) in models.items():
    for N in Ns:
        group = [
            row for row in case_rows
            if row["model"] == model and int(row["N"]) == N
        ]
        out = {"model": model, "E": E, "N": N, "I": I, "H": H, "TOPK": TOPK}
        for row in group:
            route = row["routing"]
            for key in (
                "end_to_end_ms",
                "useful_tflops",
                "overall_gpu_util_pct",
                "sm_throughput_pct",
                "sm_active_pct",
                "dram_throughput_pct",
                "l2_throughput_pct",
                "tensor_pipe_active_pct",
                "achieved_occupancy_pct",
                "ff1_gpu_util_pct",
                "ff2_gpu_util_pct",
                "bottleneck",
            ):
                out[f"{route}_{key}"] = row[key]
        out["mean_overall_gpu_util_pct"] = f"{sum(float(row['overall_gpu_util_pct']) for row in group) / len(group):.4f}"
        out["status"] = "PASS" if all(row["status"] == "PASS" for row in group) else "FAIL"
        summary_rows.append(out)

summary = root / "results_gpu_util_summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(combined, flush=True)
print(kernels, flush=True)
print(summary, flush=True)

import subprocess
from pathlib import Path


root = Path("/content/xcalibrr_topk")
for gate in ("softmax", "sigmoid"):
    result = subprocess.run(
        [
            "compute-sanitizer", "--tool", "memcheck",
            "--error-exitcode", "99",
            str(root / "build" / "benchmark"),
            "128", "8", "768", "2048", "8",
            "uniform", "0x6a09e667", "1",
            str(root / f"memcheck_{gate}.csv"), gate,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    result.check_returncode()

from __future__ import annotations

import os
from pathlib import Path
import re

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
KERNELS = HERE.parent
ARCHES = [x.strip() for x in os.environ.get(
    "XCALIBER_CUDA_ARCHS", "75,120"
).split(",") if x.strip()]
PTXAS_OPT = os.environ.get("XCALIBER_PTXAS_OPT", "3")
if PTXAS_OPT not in {"0", "1", "2", "3"}:
    raise ValueError("XCALIBER_PTXAS_OPT must be 0, 1, 2, or 3")
FF1_SYMBOL = os.environ.get("XCALIBER_FF1_SYMBOL", "")
if FF1_SYMBOL and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", FF1_SYMBOL):
    raise ValueError("XCALIBER_FF1_SYMBOL must be a C++ identifier")
GENCODE = []
for arch in ARCHES:
    GENCODE.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")


setup(
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "xcaliber_int4_vllm._C",
            ["csrc/xr64_abs_bridge.cu"],
            include_dirs=[str(KERNELS)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xptxas=-v",
                    f"-Xptxas=-O{PTXAS_OPT}",
                    *([f"-DXR64_ABS_FF1_KERNEL={FF1_SYMBOL}"]
                      if FF1_SYMBOL else []),
                    *GENCODE,
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

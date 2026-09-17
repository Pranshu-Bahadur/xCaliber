from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sources = [
    ROOT / "inference" / "fused-moe" / "bindings.cpp",
    ROOT / "inference" / "fused-moe" / "moe.cu"
]

setup(
    name="xcalibur",
    ext_modules=[
        CUDAExtension(
            name="xcalibur",
            sources=[str(s) for s in sources],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
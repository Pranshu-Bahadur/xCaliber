# xR64ABS vLLM Plugin

Self-contained Python/CUDA bridge for symmetric W4G128 MoE checkpoints.

## Supported Board

| Field | Contract |
| --- | --- |
| GPU | SM75 or SM120 |
| activation I/O | FP16 or BF16 |
| weights | symmetric INT4, group size 128 |
| routing | top-k 8 |
| local MoE parallelism | TP1 / EP1 / DP1 |
| multi-GPU | pipeline parallelism |
| activation | SiLU/SwiGLU or GELU-tanh |

The extension keeps the CUDA kernels unchanged. FP16 outer-model activations
are converted to the kernels' BF16 bit-carrier boundary inside the bridge, and
the final FP32 reduction is written back in the input dtype.

## Build

T4-only build:

```bash
XCALIBER_CUDA_ARCHS=75 python -m pip install --no-build-isolation --no-deps .
```

G4 build:

```bash
XCALIBER_CUDA_ARCHS=120 python -m pip install --no-build-isolation --no-deps .
```

The package registers `xcaliber_xr64_abs` through vLLM's general-plugin entry
point. Set `VLLM_PLUGINS=xcaliber_xr64_abs` to select it explicitly.

For two T4s, use `pipeline_parallel_size=2` and
`tensor_parallel_size=1`. This keeps every local expert tensor whole and lets
vLLM split transformer layers between devices.

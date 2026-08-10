# LLAISYS Assignment Report

## Implemented work

- Completed tensor metadata/view, slicing, contiguity, loading, copying, and device-transfer paths.
- Implemented the required CPU operators for Float32, Float16, and BFloat16.
- Added an NVIDIA CUDA runtime backend and enabled it through `xmake f --nv-gpu=y`.
- Added native CUDA kernels for add, argmax, embedding, linear, RMSNorm, RoPE, self-attention, SwiGLU, and contiguous rearrange.
- Added a native C++ Qwen2 decoder, safetensors loading through the Python ctypes wrapper, greedy generation, and a per-layer KV cache.

## Reproduction

```bash
xmake f --nv-gpu=y
xmake
xmake install
PYTHONPATH=$PWD/python python test/test_runtime.py --device nvidia
```

For an available DeepSeek-R1-Distill-Qwen-1.5B checkpoint:

```bash
PYTHONPATH=$PWD/python python test/test_infer.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B --test --device nvidia
```

## Validation results

| Platform | Status | Result |
| --- | --- | --- |
| CPU | Supported | Native Qwen2 synthetic-model generation passed. |
| NVIDIA CUDA | Supported | `test/test_runtime.py --device nvidia` passed on an RTX 5070 Ti Laptop GPU. Native CUDA operator smoke tests matched PyTorch, synthetic Qwen2 generation matched the CPU result, and the real 3.55 GB DeepSeek checkpoint loaded 339 tensors and completed single-token inference. |

The checkpoint is available locally at `F:\\Codex\\2026-08-09\\nin\\work\\DeepSeek-R1-Distill-Qwen-1.5B`. A one-token NVIDIA comparison was run: Transformers produced token `91786`, while the native path produced `0`; the end-to-end inference task therefore remains unresolved.


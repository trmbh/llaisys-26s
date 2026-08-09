from ctypes import (POINTER, Structure, c_float, c_int, c_int64, c_size_t,
                    c_void_p)
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType

try:
    from safetensors import safe_open
except ImportError:  # Keep package importable when inference dependencies are absent.
    safe_open = None


class _Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t), ("hs", c_size_t), ("nh", c_size_t),
        ("nkvh", c_size_t), ("dh", c_size_t), ("di", c_size_t),
        ("maxseq", c_size_t), ("voc", c_size_t),
        ("epsilon", c_float), ("theta", c_float), ("end_token", c_int64),
    ]


class _Weights(Structure):
    _fields_ = [
        ("in_embed", c_void_p), ("out_embed", c_void_p), ("out_norm_w", c_void_p),
        ("attn_norm_w", POINTER(c_void_p)),
        ("attn_q_w", POINTER(c_void_p)), ("attn_q_b", POINTER(c_void_p)),
        ("attn_k_w", POINTER(c_void_p)), ("attn_k_b", POINTER(c_void_p)),
        ("attn_v_w", POINTER(c_void_p)), ("attn_v_b", POINTER(c_void_p)),
        ("attn_o_w", POINTER(c_void_p)), ("attn_o_b", POINTER(c_void_p)),
        ("mlp_norm_w", POINTER(c_void_p)),
        ("mlp_gate_w", POINTER(c_void_p)), ("mlp_gate_b", POINTER(c_void_p)),
        ("mlp_up_w", POINTER(c_void_p)), ("mlp_up_b", POINTER(c_void_p)),
        ("mlp_down_w", POINTER(c_void_p)), ("mlp_down_b", POINTER(c_void_p)),
    ]


LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [POINTER(_Meta), c_int, POINTER(c_int), c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p
LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(_Weights)
LIB_LLAISYS.llaisysQwen2ModelClearCache.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelClearCache.restype = None
LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64


def _dtype_for_array(array):
    name = str(array.dtype).lower()
    if name in ("bfloat16", "bf16"):
        return DataType.BF16
    if array.dtype == np.float16:
        return DataType.F16
    if array.dtype == np.float32:
        return DataType.F32
    raise TypeError(f"Unsupported Qwen2 weight dtype: {array.dtype}")


class Qwen2:
    """Qwen2 decoder-only model backed by the native LLAISYS C++ API."""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        if safe_open is None:
            raise ImportError("Qwen2 inference requires the optional 'safetensors' package")
        model_path = Path(model_path)
        with (model_path / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        hidden = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", heads))
        head_dim = int(config.get("head_dim", hidden // heads))
        torch_dtype = str(config.get("torch_dtype", "bfloat16")).lower()
        if "float32" in torch_dtype:
            self._dtype = DataType.F32
        elif "float16" in torch_dtype:
            self._dtype = DataType.F16
        else:
            self._dtype = DataType.BF16
        self._device = DeviceType(device)
        self._end_token = int(config.get("eos_token_id", 151643))
        meta = _Meta(
            self._dtype, int(config["num_hidden_layers"]), hidden, heads,
            kv_heads, head_dim, int(config["intermediate_size"]),
            int(config.get("max_position_embeddings", 32768)),
            int(config["vocab_size"]), float(config.get("rms_norm_eps", 1e-6)),
            float(config.get("rope_theta", 10000.0)), self._end_token,
        )
        self._meta = meta
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            self._meta, int(self._device), None, 0
        )
        if not self._model:
            raise RuntimeError("Failed to create native Qwen2 model")
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        tensors = {}
        for file in sorted(model_path.glob("*.safetensors")):
            with safe_open(file, framework="numpy", device="cpu") as data:
                for name in data.keys():
                    array = np.ascontiguousarray(data.get_tensor(name))
                    actual_dtype = _dtype_for_array(array)
                    if actual_dtype != self._dtype:
                        if self._dtype == DataType.F32:
                            array = array.astype(np.float32)
                        elif self._dtype == DataType.F16:
                            array = array.astype(np.float16)
                        else:
                            raise TypeError(f"Weight {name} has {array.dtype}, expected bfloat16")
                    tensors[name] = self._load_tensor(array)

        self._assign_weights(tensors, hidden, heads, kv_heads, head_dim)

    def _load_tensor(self, array):
        dtype = self._dtype
        if dtype is None:
            dtype = _dtype_for_array(array)
        shape = (c_size_t * array.ndim)(*array.shape)
        handle = LIB_LLAISYS.tensorCreate(shape, c_size_t(array.ndim), int(dtype), int(self._device), 0)
        LIB_LLAISYS.tensorLoad(handle, array.ctypes.data_as(c_void_p))
        return handle

    def _zero_bias(self, size):
        dtype = self._dtype
        itemsize = 2 if dtype == DataType.BF16 or dtype == DataType.F16 else 4
        raw = np.zeros(size, dtype=np.uint16 if itemsize == 2 else np.float32)
        return self._load_tensor(raw)

    @staticmethod
    def _get(tensors, name):
        if name not in tensors:
            raise KeyError(f"Missing Qwen2 weight: {name}")
        return tensors[name]

    def _assign_weights(self, tensors, hidden, heads, kv_heads, head_dim):
        w = self._weights
        n = self._meta.nlayer
        w.in_embed = self._get(tensors, "model.embed_tokens.weight")
        w.out_embed = self._get(tensors, "lm_head.weight")
        w.out_norm_w = self._get(tensors, "model.norm.weight")
        qout, kvout = heads * head_dim, kv_heads * head_dim
        for i in range(n):
            prefix = f"model.layers.{i}"
            w.attn_norm_w[i] = self._get(tensors, prefix + ".input_layernorm.weight")
            for proj, out, ww, wb in (("q_proj", qout, w.attn_q_w, w.attn_q_b),
                                      ("k_proj", kvout, w.attn_k_w, w.attn_k_b),
                                      ("v_proj", kvout, w.attn_v_w, w.attn_v_b)):
                ww[i] = self._get(tensors, f"{prefix}.self_attn.{proj}.weight")
                wb[i] = tensors.get(f"{prefix}.self_attn.{proj}.bias", self._zero_bias(out))
            w.attn_o_w[i] = self._get(tensors, prefix + ".self_attn.o_proj.weight")
            w.attn_o_b[i] = self._zero_bias(hidden)
            w.mlp_norm_w[i] = self._get(tensors, prefix + ".post_attention_layernorm.weight")
            di = int(self._meta.di)
            w.mlp_gate_w[i] = self._get(tensors, prefix + ".mlp.gate_proj.weight")
            w.mlp_gate_b[i] = self._zero_bias(di)
            w.mlp_up_w[i] = self._get(tensors, prefix + ".mlp.up_proj.weight")
            w.mlp_up_b[i] = self._zero_bias(di)
            w.mlp_down_w[i] = self._get(tensors, prefix + ".mlp.down_proj.weight")
            w.mlp_down_b[i] = self._zero_bias(hidden)

    def generate(self, inputs: Sequence[int], max_new_tokens: int = None,
                 top_k: int = 1, top_p: float = 0.8, temperature: float = 0.8):
        del top_k, top_p, temperature  # Native implementation currently uses greedy argmax.
        tokens = [int(x) for x in inputs]
        steps = 128 if max_new_tokens is None else int(max_new_tokens)
        if not tokens or steps <= 0:
            return tokens
        LIB_LLAISYS.llaisysQwen2ModelClearCache(self._model)
        ids = (c_int64 * len(tokens))(*tokens)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, ids, len(tokens)))
        for _ in range(max(0, steps)):
            tokens.append(next_token)
            if next_token == self._end_token:
                break
            ids = (c_int64 * 1)(next_token)
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, ids, 1))
        return tokens

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None


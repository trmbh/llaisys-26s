#pragma once

#include "../utils.hpp"

#include <cmath>
#include <cstddef>

namespace llaisys::ops::detail {

template <typename T>
inline float read_float(const T &value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<float>(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
inline T write_float(float value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<T>(value);
    } else {
        return static_cast<T>(value);
    }
}

} // namespace llaisys::ops::detail

#define LLAISYS_DISPATCH_FLOAT(DTYPE, ...)                                                 \
    do {                                                                                    \
        switch (DTYPE) {                                                                    \
        case LLAISYS_DTYPE_F32: { using scalar_t = float; __VA_ARGS__; break; }             \
        case LLAISYS_DTYPE_F16: { using scalar_t = llaisys::fp16_t; __VA_ARGS__; break; }   \
        case LLAISYS_DTYPE_BF16: { using scalar_t = llaisys::bf16_t; __VA_ARGS__; break; }  \
        default: EXCEPTION_UNSUPPORTED_DATATYPE(DTYPE);                                    \
        }                                                                                   \
    } while (0)
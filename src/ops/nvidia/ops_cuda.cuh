#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {
void add(std::byte *out, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t n);
void argmax(int64_t *idx, std::byte *value, const std::byte *input, llaisysDataType_t dtype, size_t n);
void embedding(std::byte *out, const int64_t *index, const std::byte *weight, llaisysDataType_t dtype, size_t count, size_t width);
void linear(std::byte *out, const std::byte *input, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, size_t m, size_t k, size_t n);
void rms_norm(std::byte *out, const std::byte *input, const std::byte *weight, llaisysDataType_t dtype, size_t rows, size_t hidden, float eps);
void rope(std::byte *out, const std::byte *input, const int64_t *positions, llaisysDataType_t dtype, size_t seq, size_t heads, size_t dim, float theta);
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t dtype, size_t qlen, size_t heads, size_t kvlen, size_t kvheads, size_t dim, float scale);
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t n);
void copy(std::byte *out, const std::byte *input, size_t bytes);
} // namespace llaisys::ops::nvidia


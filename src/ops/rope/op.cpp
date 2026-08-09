#include "op.hpp"

#include "../cpu_utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_cuda.cuh"
#endif

#include <cmath>

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(in->ndim() == 3 && in->shape()[2] % 2 == 0, "rope expects [sequence, heads, even_dim]");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0] && pos_ids->dtype() == LLAISYS_DTYPE_I64, "invalid rope positions");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "rope tensors must be contiguous");
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        nvidia::rope(out->data(), in->data(), reinterpret_cast<const int64_t *>(pos_ids->data()), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
        return;
#endif
    }
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        auto out_cpu = Tensor::create(out->shape(), out->dtype());
        rope(out_cpu, in->to(LLAISYS_DEVICE_CPU), pos_ids->to(LLAISYS_DEVICE_CPU), theta);
        detail::copy_from_cpu(out, out_cpu);
        return;
    }
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids->data());
    const size_t seq = in->shape()[0], heads = in->shape()[1], dim = in->shape()[2], half = dim / 2;
    LLAISYS_DISPATCH_FLOAT(out->dtype(), {
        const auto *x = reinterpret_cast<const scalar_t *>(in->data());
        auto *y = reinterpret_cast<scalar_t *>(out->data());
        for (size_t s = 0; s < seq; ++s) for (size_t h = 0; h < heads; ++h) {
            for (size_t j = 0; j < half; ++j) {
                float angle = static_cast<float>(positions[s]) / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(dim));
                float c = std::cos(angle), sn = std::sin(angle);
                float a = detail::read_float(x[(s * heads + h) * dim + j]);
                float b = detail::read_float(x[(s * heads + h) * dim + half + j]);
                y[(s * heads + h) * dim + j] = detail::write_float<scalar_t>(a * c - b * sn);
                y[(s * heads + h) * dim + half + j] = detail::write_float<scalar_t>(b * c + a * sn);
            }
        }
    });
}
} // namespace llaisys::ops


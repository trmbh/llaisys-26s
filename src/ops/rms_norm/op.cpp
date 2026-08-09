#include "op.hpp"

#include "../cpu_utils.hpp"

#include <cmath>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(in->ndim() >= 1 && weight->ndim() == 1 && weight->shape()[0] == in->shape().back(), "rms_norm shape mismatch");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "rms_norm tensors must be contiguous");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        auto out_cpu = Tensor::create(out->shape(), out->dtype());
        rms_norm(out_cpu, in->to(LLAISYS_DEVICE_CPU), weight->to(LLAISYS_DEVICE_CPU), eps);
        detail::copy_from_cpu(out, out_cpu);
        return;
    }
    const size_t hidden = in->shape().back(), rows = in->numel() / hidden;
    LLAISYS_DISPATCH_FLOAT(out->dtype(), {
        const auto *x = reinterpret_cast<const scalar_t *>(in->data());
        const auto *w = reinterpret_cast<const scalar_t *>(weight->data());
        auto *y = reinterpret_cast<scalar_t *>(out->data());
        for (size_t r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (size_t j = 0; j < hidden; ++j) { float v = detail::read_float(x[r * hidden + j]); mean += v * v; }
            mean = 1.0f / std::sqrt(mean / static_cast<float>(hidden) + eps);
            for (size_t j = 0; j < hidden; ++j) y[r * hidden + j] = detail::write_float<scalar_t>(detail::read_float(x[r * hidden + j]) * mean * detail::read_float(w[j]));
        }
    });
}
} // namespace llaisys::ops
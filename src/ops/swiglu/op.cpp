#include "op.hpp"

#include "../cpu_utils.hpp"

#include <cmath>

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "swiglu tensors must be contiguous");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;
    LLAISYS_DISPATCH_FLOAT(out->dtype(), {
        const auto *g = reinterpret_cast<const scalar_t *>(gate->data());
        const auto *u = reinterpret_cast<const scalar_t *>(up->data());
        auto *o = reinterpret_cast<scalar_t *>(out->data());
        for (size_t i = 0; i < out->numel(); ++i) {
            float gv = detail::read_float(g[i]);
            o[i] = detail::write_float<scalar_t>(detail::read_float(u[i]) * gv / (1.0f + std::exp(-gv)));
        }
    });
}
} // namespace llaisys::ops
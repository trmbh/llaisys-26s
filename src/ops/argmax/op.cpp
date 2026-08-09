#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_utils.hpp"

#include <limits>

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1 && max_idx->numel() == 1 && max_val->numel() == 1, "argmax expects a 1D input and scalar outputs");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax index output must be int64");
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "argmax tensors must be contiguous");
    if (vals->deviceType() != LLAISYS_DEVICE_CPU) {
        auto vals_cpu = vals->to(LLAISYS_DEVICE_CPU);
        auto idx_cpu = max_idx->to(LLAISYS_DEVICE_CPU);
        auto value_cpu = max_val->to(LLAISYS_DEVICE_CPU);
        argmax(idx_cpu, value_cpu, vals_cpu);
        detail::copy_from_cpu(max_idx, idx_cpu);
        detail::copy_from_cpu(max_val, value_cpu);
        return;
    }
    auto *idx = reinterpret_cast<int64_t *>(max_idx->data());
    LLAISYS_DISPATCH_FLOAT(vals->dtype(), {
        const auto *in = reinterpret_cast<const scalar_t *>(vals->data());
        auto *out = reinterpret_cast<scalar_t *>(max_val->data());
        size_t best = 0;
        float best_value = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < vals->numel(); ++i) {
            float value = detail::read_float(in[i]);
            if (value > best_value) { best_value = value; best = i; }
        }
        *idx = static_cast<int64_t>(best);
        *out = detail::write_float<scalar_t>(best_value);
    });
}
} // namespace llaisys::ops
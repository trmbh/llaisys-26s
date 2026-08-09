#include "op.hpp"

#include "../cpu_utils.hpp"

#include <algorithm>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(weight->ndim() == 2 && index->ndim() == 1, "embedding expects a 2D weight and 1D indices");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding indices must be int64");
    const std::vector<size_t> expected_shape{index->shape()[0], weight->shape()[1]};
    CHECK_ARGUMENT(out->shape() == expected_shape, "embedding output shape mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "embedding tensors must be contiguous");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;
    const auto *indices = reinterpret_cast<const int64_t *>(index->data());
    LLAISYS_DISPATCH_FLOAT(weight->dtype(), {
        const auto *table = reinterpret_cast<const scalar_t *>(weight->data());
        auto *dst = reinterpret_cast<scalar_t *>(out->data());
        for (size_t i = 0; i < index->numel(); ++i) {
            CHECK_ARGUMENT(indices[i] >= 0 && static_cast<size_t>(indices[i]) < weight->shape()[0], "embedding index out of range");
            std::copy(table + static_cast<size_t>(indices[i]) * weight->shape()[1],
                      table + (static_cast<size_t>(indices[i]) + 1) * weight->shape()[1], dst + i * weight->shape()[1]);
        }
    });
}
} // namespace llaisys::ops
#include "op.hpp"

#include "../cpu_utils.hpp"

#include <algorithm>

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 2 && bias->ndim() == 1, "linear expects 2D input/weight and 1D bias");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1] && weight->shape()[0] == bias->shape()[0], "linear dimensions mismatch");
    const std::vector<size_t> expected_shape{in->shape()[0], weight->shape()[0]};
    CHECK_ARGUMENT(out->shape() == expected_shape, "linear output shape mismatch");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "linear tensors must be contiguous");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        auto out_cpu = Tensor::create(out->shape(), out->dtype());
        linear(out_cpu, in->to(LLAISYS_DEVICE_CPU), weight->to(LLAISYS_DEVICE_CPU), bias->to(LLAISYS_DEVICE_CPU));
        detail::copy_from_cpu(out, out_cpu);
        return;
    }
    const size_t m = in->shape()[0], k = in->shape()[1], n = weight->shape()[0];
    LLAISYS_DISPATCH_FLOAT(out->dtype(), {
        const auto *x = reinterpret_cast<const scalar_t *>(in->data());
        const auto *w = reinterpret_cast<const scalar_t *>(weight->data());
        const auto *b = reinterpret_cast<const scalar_t *>(bias->data());
        auto *y = reinterpret_cast<scalar_t *>(out->data());
        for (size_t i = 0; i < m; ++i) for (size_t j = 0; j < n; ++j) {
            float sum = detail::read_float(b[j]);
            for (size_t t = 0; t < k; ++t) sum += detail::read_float(x[i * k + t]) * detail::read_float(w[j * k + t]);
            y[i * n + j] = detail::write_float<scalar_t>(sum);
        }
    });
}
} // namespace llaisys::ops
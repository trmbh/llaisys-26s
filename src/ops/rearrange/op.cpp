#include "op.hpp"

#include "../cpu_utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_cuda.cuh"
#endif

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->shape() == in->shape(), "rearrange shape mismatch");
    ASSERT(out->isContiguous(), "rearrange output must be contiguous");
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA && in->isContiguous()) {
#ifdef ENABLE_NVIDIA_API
        nvidia::copy(out->data(), in->data(), out->numel() * out->elementSize());
        return;
#endif
    }
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        auto out_cpu = Tensor::create(out->shape(), out->dtype());
        rearrange(out_cpu, in->to(LLAISYS_DEVICE_CPU));
        detail::copy_from_cpu(out, out_cpu);
        return;
    }
    if (in->isContiguous()) {
        std::memcpy(out->data(), in->data(), in->numel() * in->elementSize());
    } else {
        auto tmp = in->contiguous();
        std::memcpy(out->data(), tmp->data(), in->numel() * in->elementSize());
    }
}
} // namespace llaisys::ops


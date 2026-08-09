#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    // always support cpu calculation
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

    auto c_cpu = Tensor::create(c->shape(), c->dtype());
    add(c_cpu, a->to(LLAISYS_DEVICE_CPU), b->to(LLAISYS_DEVICE_CPU));
    auto converted = c_cpu->to(c->deviceType(), c->deviceId());
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());
    llaisys::core::context().runtime().api()->memcpy_sync(
        c->data(), converted->data(), c->numel() * c->elementSize(), LLAISYS_MEMCPY_D2D);
    return;

    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
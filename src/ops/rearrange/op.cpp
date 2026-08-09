#include "op.hpp"

#include "../cpu_utils.hpp"

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->shape() == in->shape(), "rearrange shape mismatch");
    ASSERT(out->isContiguous(), "rearrange output must be contiguous");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;
    if (in->isContiguous()) {
        std::memcpy(out->data(), in->data(), in->numel() * in->elementSize());
    } else {
        auto tmp = in->contiguous();
        std::memcpy(out->data(), tmp->data(), in->numel() * in->elementSize());
    }
}
} // namespace llaisys::ops
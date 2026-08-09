#include "op.hpp"

#include "../cpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "self_attention expects 3D tensors");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2] && k->shape() == v->shape(), "self_attention dimensions mismatch");
    CHECK_ARGUMENT(attn_val->shape() == q->shape(), "self_attention output shape mismatch");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "query heads must be a multiple of kv heads");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "self_attention tensors must be contiguous");
    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
        auto out_cpu = Tensor::create(attn_val->shape(), attn_val->dtype());
        self_attention(out_cpu, q->to(LLAISYS_DEVICE_CPU), k->to(LLAISYS_DEVICE_CPU), v->to(LLAISYS_DEVICE_CPU), scale);
        detail::copy_from_cpu(attn_val, out_cpu);
        return;
    }
    const size_t qlen = q->shape()[0], nh = q->shape()[1], kvlen = k->shape()[0], nkvh = k->shape()[1], hd = q->shape()[2];
    const size_t group = nh / nkvh;
    LLAISYS_DISPATCH_FLOAT(attn_val->dtype(), {
        const auto *query = reinterpret_cast<const scalar_t *>(q->data());
        const auto *key = reinterpret_cast<const scalar_t *>(k->data());
        const auto *value = reinterpret_cast<const scalar_t *>(v->data());
        auto *out = reinterpret_cast<scalar_t *>(attn_val->data());
        std::vector<float> scores(kvlen);
        for (size_t qi = 0; qi < qlen; ++qi) for (size_t h = 0; h < nh; ++h) {
            const size_t kh = h / group;
            const size_t last = std::min(kvlen, kvlen - qlen + qi + 1);
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t ki = 0; ki < last; ++ki) {
                float score = 0.0f;
                for (size_t d = 0; d < hd; ++d) score += detail::read_float(query[(qi * nh + h) * hd + d]) * detail::read_float(key[(ki * nkvh + kh) * hd + d]);
                scores[ki] = score * scale;
                max_score = std::max(max_score, scores[ki]);
            }
            float denom = 0.0f;
            for (size_t ki = 0; ki < last; ++ki) { scores[ki] = std::exp(scores[ki] - max_score); denom += scores[ki]; }
            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t ki = 0; ki < last; ++ki) acc += (scores[ki] / denom) * detail::read_float(value[(ki * nkvh + kh) * hd + d]);
                out[(qi * nh + h) * hd + d] = detail::write_float<scalar_t>(acc);
            }
        }
    });
}
} // namespace llaisys::ops
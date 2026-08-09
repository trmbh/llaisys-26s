#include "llaisys/models/qwen2.h"

#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    LlaisysQwen2Weights weights{};

    explicit LlaisysQwen2Model(const LlaisysQwen2Meta &m, llaisysDeviceType_t d, int id)
        : meta(m), device(d), device_id(id) {
        const size_t n = meta.nlayer;
        weights.attn_norm_w = new llaisysTensor_t[n]{};
        weights.attn_q_w = new llaisysTensor_t[n]{};
        weights.attn_q_b = new llaisysTensor_t[n]{};
        weights.attn_k_w = new llaisysTensor_t[n]{};
        weights.attn_k_b = new llaisysTensor_t[n]{};
        weights.attn_v_w = new llaisysTensor_t[n]{};
        weights.attn_v_b = new llaisysTensor_t[n]{};
        weights.attn_o_w = new llaisysTensor_t[n]{};
        weights.attn_o_b = new llaisysTensor_t[n]{};
        weights.mlp_norm_w = new llaisysTensor_t[n]{};
        weights.mlp_gate_w = new llaisysTensor_t[n]{};
        weights.mlp_gate_b = new llaisysTensor_t[n]{};
        weights.mlp_up_w = new llaisysTensor_t[n]{};
        weights.mlp_up_b = new llaisysTensor_t[n]{};
        weights.mlp_down_w = new llaisysTensor_t[n]{};
        weights.mlp_down_b = new llaisysTensor_t[n]{};
    }

    ~LlaisysQwen2Model() {
        auto destroy = [](llaisysTensor_t t) {
            if (t != nullptr) delete t;
        };
        destroy(weights.in_embed);
        destroy(weights.out_embed);
        destroy(weights.out_norm_w);
        const size_t n = meta.nlayer;
        for (size_t i = 0; i < n; ++i) {
            destroy(weights.attn_norm_w[i]);
            destroy(weights.attn_q_w[i]); destroy(weights.attn_q_b[i]);
            destroy(weights.attn_k_w[i]); destroy(weights.attn_k_b[i]);
            destroy(weights.attn_v_w[i]); destroy(weights.attn_v_b[i]);
            destroy(weights.attn_o_w[i]); destroy(weights.attn_o_b[i]);
            destroy(weights.mlp_norm_w[i]);
            destroy(weights.mlp_gate_w[i]); destroy(weights.mlp_gate_b[i]);
            destroy(weights.mlp_up_w[i]); destroy(weights.mlp_up_b[i]);
            destroy(weights.mlp_down_w[i]); destroy(weights.mlp_down_b[i]);
        }
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w; delete[] weights.attn_q_b;
        delete[] weights.attn_k_w; delete[] weights.attn_k_b;
        delete[] weights.attn_v_w; delete[] weights.attn_v_b;
        delete[] weights.attn_o_w; delete[] weights.attn_o_b; delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w; delete[] weights.mlp_gate_b;
        delete[] weights.mlp_up_w; delete[] weights.mlp_up_b;
        delete[] weights.mlp_down_w; delete[] weights.mlp_down_b;
    }
};

namespace {
using namespace llaisys;

tensor_t make_tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype,
                     llaisysDeviceType_t device, int device_id) {
    return Tensor::create(shape, dtype, device, device_id);
}

void add_inplace(tensor_t dst, tensor_t a, tensor_t b) {
    ops::add(dst, a, b);
}

} // namespace

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                llaisysDeviceType_t device,
                                                int *device_ids, int ndevice) {
        (void)ndevice;
        const int device_id = device_ids == nullptr ? 0 : device_ids[0];
        return new LlaisysQwen2Model(*meta, device, device_id);
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) { delete model; }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return &model->weights;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        CHECK_ARGUMENT(model != nullptr && token_ids != nullptr && ntoken > 0, "invalid Qwen2 inference input");
        const auto &m = model->meta;
        const auto dtype = m.dtype;
        const auto dev = model->device;
        const int did = model->device_id;
        auto ids = make_tensor({ntoken}, LLAISYS_DTYPE_I64, dev, did);
        ids->load(token_ids);
        auto hidden = make_tensor({ntoken, m.hs}, dtype, dev, did);
        ops::embedding(hidden, ids, model->weights.in_embed->tensor);
        auto pos = make_tensor({ntoken}, LLAISYS_DTYPE_I64, dev, did);
        std::vector<int64_t> positions(ntoken);
        for (size_t i = 0; i < ntoken; ++i) positions[i] = static_cast<int64_t>(i);
        pos->load(positions.data());

        const std::vector<size_t> qshape{ntoken, m.nh, m.dh};
        const std::vector<size_t> kvshape{ntoken, m.nkvh, m.dh};
        const float scale = 1.0f / std::sqrt(static_cast<float>(m.dh));
        for (size_t layer = 0; layer < m.nlayer; ++layer) {
            auto norm = make_tensor({ntoken, m.hs}, dtype, dev, did);
            ops::rms_norm(norm, hidden, model->weights.attn_norm_w[layer]->tensor, m.epsilon);
            auto qflat = make_tensor({ntoken, m.nh * m.dh}, dtype, dev, did);
            auto kflat = make_tensor({ntoken, m.nkvh * m.dh}, dtype, dev, did);
            auto vflat = make_tensor({ntoken, m.nkvh * m.dh}, dtype, dev, did);
            ops::linear(qflat, norm, model->weights.attn_q_w[layer]->tensor, model->weights.attn_q_b[layer]->tensor);
            ops::linear(kflat, norm, model->weights.attn_k_w[layer]->tensor, model->weights.attn_k_b[layer]->tensor);
            ops::linear(vflat, norm, model->weights.attn_v_w[layer]->tensor, model->weights.attn_v_b[layer]->tensor);
            auto q = qflat->view(qshape);
            auto k = kflat->view(kvshape);
            auto v = vflat->view(kvshape);
            auto qrot = make_tensor(qshape, dtype, dev, did);
            auto krot = make_tensor(kvshape, dtype, dev, did);
            ops::rope(qrot, q, pos, m.theta);
            ops::rope(krot, k, pos, m.theta);
            auto attn = make_tensor(qshape, dtype, dev, did);
            ops::self_attention(attn, qrot, krot, v, scale);
            auto attn2 = attn->view({ntoken, m.hs});
            auto proj = make_tensor({ntoken, m.hs}, dtype, dev, did);
            ops::linear(proj, attn2, model->weights.attn_o_w[layer]->tensor,
                        model->weights.attn_o_b[layer]->tensor);
            auto residual = make_tensor({ntoken, m.hs}, dtype, dev, did);
            add_inplace(residual, hidden, proj);
            auto post_norm = make_tensor({ntoken, m.hs}, dtype, dev, did);
            ops::rms_norm(post_norm, residual, model->weights.mlp_norm_w[layer]->tensor, m.epsilon);
            auto gate = make_tensor({ntoken, m.di}, dtype, dev, did);
            auto up = make_tensor({ntoken, m.di}, dtype, dev, did);
            ops::linear(gate, post_norm, model->weights.mlp_gate_w[layer]->tensor, model->weights.mlp_gate_b[layer]->tensor);
            ops::linear(up, post_norm, model->weights.mlp_up_w[layer]->tensor, model->weights.mlp_up_b[layer]->tensor);
            auto activated = make_tensor({ntoken, m.di}, dtype, dev, did);
            ops::swiglu(activated, gate, up);
            auto down = make_tensor({ntoken, m.hs}, dtype, dev, did);
            ops::linear(down, activated, model->weights.mlp_down_w[layer]->tensor, model->weights.mlp_down_b[layer]->tensor);
            hidden = make_tensor({ntoken, m.hs}, dtype, dev, did);
            add_inplace(hidden, residual, down);
        }
        auto final_norm = make_tensor({ntoken, m.hs}, dtype, dev, did);
        ops::rms_norm(final_norm, hidden, model->weights.out_norm_w->tensor, m.epsilon);
        auto logits = make_tensor({ntoken, m.voc}, dtype, dev, did);
        auto out_bias = make_tensor({m.voc}, dtype, dev, did);
        std::vector<uint8_t> zeros(out_bias->numel() * out_bias->elementSize(), 0);
        out_bias->load(zeros.data());
        ops::linear(logits, final_norm, model->weights.out_embed->tensor, out_bias);
        auto last = logits->slice(0, ntoken - 1, ntoken)->view({m.voc});
        auto idx = make_tensor({1}, LLAISYS_DTYPE_I64, dev, did);
        auto val = make_tensor({1}, dtype, dev, did);
        ops::argmax(idx, val, last);
        auto idx_cpu = idx->to(LLAISYS_DEVICE_CPU);
        return reinterpret_cast<const int64_t *>(idx_cpu->data())[0];
    }
}


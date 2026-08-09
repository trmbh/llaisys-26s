#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <functional>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    ptrdiff_t expected = 1;
    for (size_t i = ndim(); i-- > 0;) {
        if (_meta.shape[i] > 1 && _meta.strides[i] != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(_meta.shape[i]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "permute order must have the same rank");
    std::vector<bool> seen(ndim(), false);
    TensorMeta meta = _meta;
    meta.shape.resize(ndim());
    meta.strides.resize(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        CHECK_ARGUMENT(order[i] < ndim() && !seen[order[i]], "invalid permute order");
        seen[order[i]] = true;
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(isContiguous(), "view requires a contiguous tensor");
    size_t old_numel = numel();
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(old_numel == new_numel, "view shape has a different number of elements");
    std::vector<ptrdiff_t> strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }
    return std::shared_ptr<Tensor>(new Tensor(TensorMeta{_meta.dtype, shape, strides}, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice dimension out of range");
    CHECK_ARGUMENT(start <= end && end <= _meta.shape[dim], "slice range out of range");
    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;
    size_t offset = _offset + static_cast<size_t>(start * _meta.strides[dim]) * elementSize();
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr || numel() == 0, "source pointer is null");
    core::context().setDevice(deviceType(), deviceId());
    auto kind = deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(data(), src_, numel() * elementSize(), kind);
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    auto out = create(_meta.shape, _meta.dtype, deviceType(), deviceId());
    core::context().setDevice(deviceType(), deviceId());
    auto *api = core::context().runtime().api();
    const auto kind = deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
    std::vector<size_t> index(ndim(), 0);
    size_t linear = 0;
    std::function<void(size_t, size_t)> copy_dim = [&](size_t dim, size_t src_elem_offset) {
        if (dim == ndim()) {
            api->memcpy_sync(out->data() + linear * elementSize(), data() + src_elem_offset * elementSize(), elementSize(), kind);
            ++linear;
            return;
        }
        for (size_t i = 0; i < _meta.shape[dim]; ++i) {
            copy_dim(dim + 1, src_elem_offset + i * static_cast<size_t>(_meta.strides[dim]));
        }
    };
    copy_dim(0, 0);
    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device < 0) {
        device = deviceType() == device_type ? deviceId() : 0;
    }
    auto src = isContiguous() ? std::shared_ptr<const Tensor>(this, [](const Tensor *) {}) : std::shared_ptr<const Tensor>(contiguous());
    auto out = create(_meta.shape, _meta.dtype, device_type, device);
    core::context().setDevice(device_type, device);
    auto *api = core::context().runtime().api();
    llaisysMemcpyKind_t kind;
    if (deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_H2H;
    else if (deviceType() == LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_H2D;
    else if (device_type == LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_D2H;
    else kind = LLAISYS_MEMCPY_D2D;
    api->memcpy_sync(out->data(), src->data(), numel() * elementSize(), kind);
    return out;
}

} // namespace llaisys
#include "ops_cuda.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace llaisys::ops::nvidia {
namespace {
constexpr int kThreads = 256;

inline void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__device__ float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { int e = -14; while ((mant & 0x400) == 0) { mant <<= 1; --e; } bits = sign | uint32_t(e + 127) << 23 | ((mant & 0x3ff) << 13); }
    } else if (exp == 31) bits = sign | 0x7f800000 | (mant << 13);
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    return __uint_as_float(bits);
}
__device__ uint16_t float_to_half(float x) {
    const uint32_t b = __float_as_uint(x), sign = (b >> 16) & 0x8000;
    int exp = int((b >> 23) & 0xff) - 127;
    if (exp >= 16) return uint16_t(sign | 0x7c00);
    if (exp < -24) return uint16_t(sign);
    uint32_t mant = (b & 0x7fffff) | (exp < -14 ? 0x800000 : 0);
    if (exp < -14) mant >>= (-14 - exp);
    return uint16_t(sign | ((exp < -14 ? 0 : exp + 15) << 10) | (mant >> 13));
}
__device__ float load(const std::byte *p, llaisysDataType_t type, size_t i) {
    if (type == LLAISYS_DTYPE_F32) return reinterpret_cast<const float *>(p)[i];
    uint16_t bits = reinterpret_cast<const uint16_t *>(p)[i];
    return type == LLAISYS_DTYPE_F16 ? half_to_float(bits) : __uint_as_float(uint32_t(bits) << 16);
}
__device__ void store(std::byte *p, llaisysDataType_t type, size_t i, float x) {
    if (type == LLAISYS_DTYPE_F32) reinterpret_cast<float *>(p)[i] = x;
    else reinterpret_cast<uint16_t *>(p)[i] = type == LLAISYS_DTYPE_F16 ? float_to_half(x) : uint16_t(__float_as_uint(x) >> 16);
}
__global__ void add_k(std::byte *o,const std::byte*a,const std::byte*b,llaisysDataType_t t,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)store(o,t,i,load(a,t,i)+load(b,t,i));}
__global__ void embed_k(std::byte*o,const int64_t*id,const std::byte*w,llaisysDataType_t t,size_t n,size_t d){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){size_t r=i/d;store(o,t,i,load(w,t,size_t(id[r])*d+i%d));}}
__global__ void linear_k(std::byte*o,const std::byte*x,const std::byte*w,const std::byte*b,llaisysDataType_t t,size_t m,size_t k,size_t n){size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z<m*n){size_t r=z/n,c=z%n;float s=load(b,t,c);for(size_t j=0;j<k;++j)s+=load(x,t,r*k+j)*load(w,t,c*k+j);store(o,t,z,s);}}
__global__ void rms_k(std::byte*o,const std::byte*x,const std::byte*w,llaisysDataType_t t,size_t rows,size_t d,float eps){size_t r=blockIdx.x;if(r>=rows)return;extern __shared__ float s[];float sum=0;for(size_t j=threadIdx.x;j<d;j+=blockDim.x){float v=load(x,t,r*d+j);sum+=v*v;}s[threadIdx.x]=sum;__syncthreads();for(int off=blockDim.x/2;off;off/=2){if(threadIdx.x<off)s[threadIdx.x]+=s[threadIdx.x+off];__syncthreads();}float inv=rsqrtf(s[0]/d+eps);for(size_t j=threadIdx.x;j<d;j+=blockDim.x)store(o,t,r*d+j,load(x,t,r*d+j)*inv*load(w,t,j));}
__global__ void rope_k(std::byte*o,const std::byte*x,const int64_t*p,llaisysDataType_t t,size_t seq,size_t h,size_t d,float theta){size_t z=blockIdx.x*blockDim.x+threadIdx.x,half=d/2;if(z<seq*h*half){size_t j=z%half,tmp=z/half,head=tmp%h,s=tmp/h,base=(s*h+head)*d;float a=load(x,t,base+j),b=load(x,t,base+half+j),ang=float(p[s])/powf(theta,2.f*j/d),c=cosf(ang),sn=sinf(ang);store(o,t,base+j,a*c-b*sn);store(o,t,base+half+j,b*c+a*sn);}}
__global__ void swiglu_k(std::byte*o,const std::byte*g,const std::byte*u,llaisysDataType_t t,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=load(g,t,i);store(o,t,i,load(u,t,i)*v/(1.f+expf(-v)));}}
__global__ void attn_k(std::byte*o,const std::byte*q,const std::byte*k,const std::byte*v,llaisysDataType_t t,size_t qlen,size_t h,size_t kvlen,size_t kh,size_t d,float scale){size_t z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=qlen*h*d)return;size_t col=z%d,tmp=z/d,head=tmp%h,qi=tmp/h,group=h/kh,khead=head/group,last=kvlen-qlen+qi+1;float mx=-INFINITY;for(size_t ki=0;ki<last;++ki){float s=0;for(size_t j=0;j<d;++j)s+=load(q,t,(qi*h+head)*d+j)*load(k,t,(ki*kh+khead)*d+j);mx=fmaxf(mx,s*scale);}float den=0,num=0;for(size_t ki=0;ki<last;++ki){float s=0;for(size_t j=0;j<d;++j)s+=load(q,t,(qi*h+head)*d+j)*load(k,t,(ki*kh+khead)*d+j);float e=expf(s*scale-mx);den+=e;num+=e*load(v,t,(ki*kh+khead)*d+col);}store(o,t,z,num/den);}
__global__ void argmax_k(int64_t*idx,std::byte*val,const std::byte*x,llaisysDataType_t t,size_t n){__shared__ float sv[kThreads];__shared__ int si[kThreads];int tid=threadIdx.x;float best=-INFINITY;int bi=0;for(size_t i=tid;i<n;i+=blockDim.x){float v=load(x,t,i);if(v>best){best=v;bi=int(i);}}sv[tid]=best;si[tid]=bi;__syncthreads();for(int off=blockDim.x/2;off;off/=2){if(tid<off&&sv[tid+off]>sv[tid]){sv[tid]=sv[tid+off];si[tid]=si[tid+off];}__syncthreads();}if(tid==0){*idx=si[0];store(val,t,0,sv[0]);}}
}
void add(std::byte*o,const std::byte*a,const std::byte*b,llaisysDataType_t t,size_t n){add_k<<<(n+kThreads-1)/kThreads,kThreads>>>(o,a,b,t,n);check(cudaGetLastError());}
void embedding(std::byte*o,const int64_t*i,const std::byte*w,llaisysDataType_t t,size_t c,size_t d){embed_k<<<(c*d+kThreads-1)/kThreads,kThreads>>>(o,i,w,t,c*d,d);check(cudaGetLastError());}
void linear(std::byte*o,const std::byte*x,const std::byte*w,const std::byte*b,llaisysDataType_t t,size_t m,size_t k,size_t n){linear_k<<<(m*n+kThreads-1)/kThreads,kThreads>>>(o,x,w,b,t,m,k,n);check(cudaGetLastError());}
void rms_norm(std::byte*o,const std::byte*x,const std::byte*w,llaisysDataType_t t,size_t r,size_t d,float e){rms_k<<<r,kThreads,kThreads*sizeof(float)>>>(o,x,w,t,r,d,e);check(cudaGetLastError());}
void rope(std::byte*o,const std::byte*x,const int64_t*p,llaisysDataType_t t,size_t s,size_t h,size_t d,float th){rope_k<<<(s*h*(d/2)+kThreads-1)/kThreads,kThreads>>>(o,x,p,t,s,h,d,th);check(cudaGetLastError());}
void self_attention(std::byte*o,const std::byte*q,const std::byte*k,const std::byte*v,llaisysDataType_t t,size_t ql,size_t h,size_t kl,size_t kh,size_t d,float sc){attn_k<<<(ql*h*d+kThreads-1)/kThreads,kThreads>>>(o,q,k,v,t,ql,h,kl,kh,d,sc);check(cudaGetLastError());}
void swiglu(std::byte*o,const std::byte*g,const std::byte*u,llaisysDataType_t t,size_t n){swiglu_k<<<(n+kThreads-1)/kThreads,kThreads>>>(o,g,u,t,n);check(cudaGetLastError());}
void copy(std::byte*o,const std::byte*i,size_t bytes){check(cudaMemcpy(o,i,bytes,cudaMemcpyDeviceToDevice));}
void argmax(int64_t*i,std::byte*v,const std::byte*x,llaisysDataType_t t,size_t n){argmax_k<<<1,kThreads>>>(i,v,x,t,n);check(cudaGetLastError());}
} // namespace llaisys::ops::nvidia


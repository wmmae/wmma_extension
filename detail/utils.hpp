#ifndef __WMMAE_UTILS_HPP__
#define __WMMAE_UTILS_HPP__
#include <cuda_fp16.h>

namespace mtk {
namespace detail {
namespace utils {
template <class T> inline __device__ T cast(const float v);
template <class T> inline __device__ T cast(const half v);
template <> inline __device__ float cast(const float v){return v;}
template <> inline __device__ float cast(const half v){return __half2float(v);}
template <> inline __device__ half cast(const float v){return __float2half(v);}
template <> inline __device__ half cast(const half v){return v;}
} // namespace utils
} // namespace detail
} // namespace mtk

#endif /* end of include guard */

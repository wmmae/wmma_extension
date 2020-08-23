#ifndef __WMMAE_DETAIL_COMMON__
#define __WMMAE_DETAIL_COMMON__
#include <cuda_fp16.h>
namespace mtk {
namespace wmma {
namespace detail {
namespace common {
template <class T> inline __device__ T cast(const float v);
template <class T> inline __device__ T cast(const half v);
template <> inline __device__ float cast(const float v){return v;}
template <> inline __device__ float cast(const half v){return __half2float(v);}
template <> inline __device__ half cast(const float v){return __float2half(v);}
template <> inline __device__ half cast(const half v){return v;}
} // namespace common
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

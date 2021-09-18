#ifndef __WMMAE_TCEC_DETAIL_SCALE_HPP__
#define __WMMAE_TCEC_DETAIL_SCALE_HPP__

namespace mtk {
namespace wmma {
namespace tcec {
namespace detail {
template <class T>
__device__ inline float correction_scale_0(const float v) {return v;}
template <>
__device__ inline float correction_scale_0<half>(const float v) {return v * 2048;}

template <class T>
__device__ inline float correction_scale_1(const float v) {return v;}
template <>
__device__ inline float correction_scale_1<half>(const float v) {return v / 2048;}
} // namespace detail
} // namespace tcec
} // namespace wmma
} // namespace mtk

#endif

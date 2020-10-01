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
inline __device__ unsigned get_lane_id() {
	unsigned lane_id;
	asm(R"({mov.s32 %0, %laneid;})":"=r"(lane_id));
	return lane_id;
}
} // namespace common
} // namespace detail

template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, float, Layout>& frag) {
	int4* const i4 = reinterpret_cast<int4*>(frag.x);
	const unsigned size = sizeof(float) * nvcuda::wmma::fragment<Use, M, N, K, float, Layout>::num_elements;
	for (unsigned i = 0; i < size / sizeof(int4); i++) {
		i4[i] = make_int4(0, 0, 0, 0);
	}
}
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& frag) {
	int4* const i4 = reinterpret_cast<int4*>(frag.x);
	const unsigned size = sizeof(half) * nvcuda::wmma::fragment<Use, M, N, K, half, Layout>::num_elements;
	for (unsigned i = 0; i < size / sizeof(int4); i++) {
		i4[i] = make_int4(0, 0, 0, 0);
	}
}
} // namespace wmma
} // namespace mtk
#endif

#ifndef __WMMAE_DETAIL_COMMON__
#define __WMMAE_DETAIL_COMMON__
#include <cuda_fp16.h>
namespace mtk {
namespace wmma {
namespace detail {
namespace common {
template <class T>
struct storage_t {using type = T;};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <> struct storage_t<nvcuda::wmma::precision::tf32> {using type = float;};
__device__ inline float to_tf32(const float a) {
	float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(a));
    return ret;
}
#endif
template <class T> inline __device__ __host__ typename storage_t<T>::type cast(const float v);
template <class T> inline __device__ __host__ typename storage_t<T>::type cast(const half v);
template <> inline __device__ __host__ typename storage_t<float>::type cast<float>(const float v){return v;}
template <> inline __device__ __host__ typename storage_t<half >::type cast<half >(const float v){return __float2half(v);}
template <> inline __device__ __host__ typename storage_t<float>::type cast<float>(const half v){return __half2float(v);}
template <> inline __device__ __host__ typename storage_t<half >::type cast<half >(const half v){return v;}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <> inline __device__ __host__ typename storage_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const float v){return to_tf32(v);}
template <> inline __device__ __host__ typename storage_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const half  v){return to_tf32(__half2float(v));}
#endif
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, nvcuda::wmma::precision::tf32, Layout>& frag) {
	int4* const i4 = reinterpret_cast<int4*>(frag.x);
	const unsigned size = sizeof(float) * nvcuda::wmma::fragment<Use, M, N, K, nvcuda::wmma::precision::tf32, Layout>::num_elements;
	for (unsigned i = 0; i < size / sizeof(int4); i++) {
		i4[i] = make_int4(0, 0, 0, 0);
	}
}
#endif
} // namespace wmma
} // namespace mtk
#endif

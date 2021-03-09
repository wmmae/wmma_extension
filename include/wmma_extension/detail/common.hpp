#ifndef __WMMAE_DETAIL_COMMON__
#define __WMMAE_DETAIL_COMMON__
#include <cuda_fp16.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
namespace nvcuda {
namespace wmma {
namespace precision {
class tf32;
} // namespace precision
} // namespace wmma
} // namespace nvcuda
#endif

namespace mtk {
namespace wmma {

namespace mma {
template <typename T, int size>
struct __align__(4) __frag_base {
	T x[size];
	enum {num_elements = size};
};

template <class T>
__device__ inline void fill_fragment(__frag_base<half, 8>& f, const T v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = v;
}
template <class T>
__device__ inline void fill_fragment(__frag_base<half, 4>& f, const T v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = v;
}
template <class T>
__device__ inline void fill_fragment(__frag_base<float, 4>& f, const T v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = v;
}

template <class Use, int m, int n, int k, class T, class Layout = void>
class fragment;
} // namespace mma

namespace detail {
namespace common {
template <class T>
struct storage_t {using type = T;};
template <class T> inline __device__ __host__ typename storage_t<T>::type cast(const float v);
template <class T> inline __device__ __host__ typename storage_t<T>::type cast(const half v);
template <> inline __device__ __host__ typename storage_t<float>::type cast<float>(const float v){return v;}
template <> inline __device__ __host__ typename storage_t<half >::type cast<half >(const float v){return __float2half(v);}
template <> inline __device__ __host__ typename storage_t<float>::type cast<float>(const half v){return __half2float(v);}
template <> inline __device__ __host__ typename storage_t<half >::type cast<half >(const half v){return v;}

template <> struct storage_t<nvcuda::wmma::precision::tf32> {using type = float;};
__device__ __host__ inline float to_tf32(const float a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(a));
    return ret;
#else
	return a;
#endif
}
template <> inline __device__ __host__ typename storage_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const float v){return to_tf32(v);}
template <> inline __device__ __host__ typename storage_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const half  v){return to_tf32(__half2float(v));}

inline __device__ unsigned get_lane_id() {
	unsigned lane_id;
	asm(R"({mov.s32 %0, %laneid;})":"=r"(lane_id));
	return lane_id;
}

template <class Use, int M, int N, int K> struct get_M;
template <int M, int N, int K> struct get_M<nvcuda::wmma::matrix_a   , M, N, K>{static const int value = M;};
template <int M, int N, int K> struct get_M<nvcuda::wmma::matrix_b   , M, N, K>{static const int value = K;};
template <int M, int N, int K> struct get_M<nvcuda::wmma::accumulator, M, N, K>{static const int value = M;};

template <class Use, int M, int N, int K> struct get_N;
template <int M, int N, int K> struct get_N<nvcuda::wmma::matrix_a   , M, N, K>{static const int value = K;};
template <int M, int N, int K> struct get_N<nvcuda::wmma::matrix_b   , M, N, K>{static const int value = N;};
template <int M, int N, int K> struct get_N<nvcuda::wmma::accumulator, M, N, K>{static const int value = N;};

template <class Layout, int col_value, int row_calue> struct layout_switch;
template <int col_value, int row_calue> struct layout_switch<nvcuda::wmma::col_major, col_value, row_value> {static const int value = col_value;};
template <int col_value, int row_calue> struct layout_switch<nvcuda::wmma::row_major, col_value, row_value> {static const int value = row_value;};

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

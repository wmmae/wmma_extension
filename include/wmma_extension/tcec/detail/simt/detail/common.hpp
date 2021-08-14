#ifndef __WMMAE_SIMT_DETAIL_COMMON__
#define __WMMAE_SIMT_DETAIL_COMMON__
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
namespace mma_simt {
template <class Use, int m, int n, int k, class T, class Layout = void>
class fragment;

namespace detail {
template <typename T, int size>
struct __align__(4) __frag_base {
	T x[size];
	enum {num_elements = size};
};
template <int size>
struct __align__(2) __frag_base<half, size> {
	half x[size];
	enum {num_elements = size};
};
template <int size>
struct __align__(8) __frag_base<double, size> {
	double x[size];
	enum {num_elements = size};
};

template <class Use, int M, int N, int K> struct select_value;
template <int M, int N, int K> struct select_value<nvcuda::wmma::matrix_a   , M, N, K>{static const int value = M;};
template <int M, int N, int K> struct select_value<nvcuda::wmma::matrix_b   , M, N, K>{static const int value = N;};
template <int M, int N, int K> struct select_value<nvcuda::wmma::accumulator, M, N, K>{static const int value = K;};

template <class Use, int M, int N, int K> struct get_M : public select_value<Use, M, K, M>{};
template <class Use, int M, int N, int K> struct get_N : public select_value<Use, K, N, N>{};

template <class Layout, int col_value, int row_value> struct layout_switch;
template <int col_value, int row_value> struct layout_switch<nvcuda::wmma::col_major, col_value, row_value> {static const int value = col_value;};
template <int col_value, int row_value> struct layout_switch<nvcuda::wmma::row_major, col_value, row_value> {static const int value = row_value;};

template <class T>
struct storage_t {using type = T;};
template <class DST, class SRC> inline __device__ __host__ typename storage_t<DST>::type cast(const SRC v) {return static_cast<DST>(v);}
template <> inline __device__ __host__ typename storage_t<float>::type cast<float, half >(const half  v) {return __half2float(v);}
template <> inline __device__ __host__ typename storage_t<half>::type  cast<half , float>(const float v) {return __float2half(v);}

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
template <> inline __device__ __host__ typename storage_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32, float>(const float v){return to_tf32(v);}

} // namespace detail

template <class T, class S, int size>
__device__ inline void fill_fragment(detail::__frag_base<T, size>& f, const S v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = detail::cast<T>(v);
}

template <class T, int size>
__device__ inline void fill_zero(detail::__frag_base<T, size>& f) {
	fill_fragment(f, 0.0f);
}

} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif

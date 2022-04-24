#ifndef __WMMAE_DETAIL_COMMON__
#define __WMMAE_DETAIL_COMMON__
#include <cstdint>
#include <mma.h>
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

namespace detail {
template <unsigned byte, class T>
struct fill_zero_core;

template <class T>
struct fill_zero_core<2, T> {
	__device__ void operator()(T* const ptr) {
		*reinterpret_cast<uint16_t*>(ptr) = 0;
	}
};

template <class T>
struct fill_zero_core<4, T> {
	__device__ void operator()(T* const ptr) {
		*reinterpret_cast<uint32_t*>(ptr) = 0;
	}
};

template <class T>
struct fill_zero_core<8, T> {
	__device__ void operator()(T* const ptr) {
		*reinterpret_cast<int2*>(ptr) = make_int2(0, 0);
	}
};

template <class T>
struct fill_zero_core<16, T> {
	__device__ void operator()(T* const ptr) {
		*reinterpret_cast<int4*>(ptr) = make_int4(0, 0, 0, 0);
	}
};

template <class T>
struct fill_zero_core<32, T> {
	__device__ void operator()(T* const ptr) {
		*(reinterpret_cast<int4*>(ptr) + 0) = make_int4(0, 0, 0, 0);
		*(reinterpret_cast<int4*>(ptr) + 1) = make_int4(0, 0, 0, 0);
	}
};

template <class T>
struct fill_zero_core<64, T> {
	__device__ void operator()(T* const ptr) {
		*(reinterpret_cast<int4*>(ptr) + 0) = make_int4(0, 0, 0, 0);
		*(reinterpret_cast<int4*>(ptr) + 1) = make_int4(0, 0, 0, 0);
		*(reinterpret_cast<int4*>(ptr) + 2) = make_int4(0, 0, 0, 0);
		*(reinterpret_cast<int4*>(ptr) + 3) = make_int4(0, 0, 0, 0);
	}
};
} // namespace detail

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
__device__ inline void fill_fragment(__frag_base<half, 2>& f, const T v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = v;
}
template <class T>
__device__ inline void fill_fragment(__frag_base<float, 2>& f, const T v) {
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

template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(mtk::wmma::mma::fragment<Use, M, N, K, float, Layout>& frag) {
	constexpr unsigned size = 4 * mtk::wmma::mma::fragment<Use, M, N, K, float, Layout>::num_elements;
	detail::fill_zero_core<size, float>{}(reinterpret_cast<float*>(frag.x));
}

template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(mtk::wmma::mma::fragment<Use, M, N, K, nvcuda::wmma::precision::tf32, Layout>& frag) {
	constexpr unsigned size = 4 * mtk::wmma::mma::fragment<Use, M, N, K, nvcuda::wmma::precision::tf32, Layout>::num_elements;
	detail::fill_zero_core<size, float>{}(reinterpret_cast<float*>(frag.x));
}

template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(mtk::wmma::mma::fragment<Use, M, N, K, half, Layout>& frag) {
	constexpr unsigned size = 2 * mtk::wmma::mma::fragment<Use, M, N, K, half, Layout>::num_elements;
	detail::fill_zero_core<size, half>{}(reinterpret_cast<half*>(frag.x));
}
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

template <class Layout, int col_value, int row_value> struct layout_switch;
template <int col_value, int row_value> struct layout_switch<nvcuda::wmma::col_major, col_value, row_value> {static const int value = col_value;};
template <int col_value, int row_value> struct layout_switch<nvcuda::wmma::row_major, col_value, row_value> {static const int value = row_value;};

} // namespace common
} // namespace detail
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, float, Layout>& frag) {
	const unsigned size = 4 * nvcuda::wmma::fragment<Use, M, N, K, float, Layout>::num_elements;
	detail::fill_zero_core<size, float>{}(reinterpret_cast<float*>(frag.x));
}
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& frag) {
	const unsigned size = 2 * nvcuda::wmma::fragment<Use, M, N, K, half, Layout>::num_elements;
	detail::fill_zero_core<size, half>{}(reinterpret_cast<half*>(frag.x));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, nvcuda::wmma::precision::tf32, Layout>& frag) {
	const unsigned size = 4 * nvcuda::wmma::fragment<Use, M, N, K, nvcuda::wmma::precision::tf32, Layout>::num_elements;
	detail::fill_zero_core<size, float>{}(reinterpret_cast<float*>(frag.x));
}
#endif
} // namespace wmma
} // namespace mtk
#endif

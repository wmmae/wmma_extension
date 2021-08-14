#ifndef __MTK_HMMA_F32_F32_DETAIL_HPP__
#define __MTK_HMMA_F32_F32_DETAIL_HPP__
#include <mma.h>
#include <type_traits>
#include <cuda_fp16.h>
#include "wmma_extension_include.hpp"

namespace mtk {
namespace wmma {
namespace tcec {
namespace detail {
template <class Use, int a, int b, int c>
struct select_value {};
template <int a, int b, int c>
struct select_value<nvcuda::wmma::matrix_a   , a, b, c> {const static int value = a;};
template <int a, int b, int c>
struct select_value<nvcuda::wmma::matrix_b   , a, b, c> {const static int value = b;};
template <int a, int b, int c>
struct select_value<nvcuda::wmma::accumulator, a, b, c> {const static int value = c;};

template <class T>
__device__ constexpr int get_fragment_k() {return 16;};
template <> __device__ constexpr int get_fragment_k<nvcuda::wmma::precision::tf32>() {return 8 ;}

template <int frag_m, int frag_n, class Layout>
struct compute_mem_offset {
	// calculate memory ofset from mem_index given by foreach
	__device__ unsigned operator()(const unsigned mem_offset, const unsigned ldm, const unsigned m_offset, const unsigned n_offset) {
		return ((mem_offset % frag_n) + n_offset) + (mem_offset / frag_n + m_offset) * ldm;
	}
	// calculate memory ofset from matrix position (i,j) given by foreach_ij
	__device__ unsigned operator()(const unsigned i, const unsigned j, const unsigned ldm, const unsigned m_offset, const unsigned n_offset) {
		return (j + n_offset) + (i + m_offset) * ldm;
	}
};

template <int frag_m, int frag_n>
struct compute_mem_offset<frag_m, frag_n, nvcuda::wmma::col_major> {
	// calculate memory ofset from mem_index given by foreach
	__device__ unsigned operator()(const unsigned mem_offset, const unsigned ldm, const unsigned m_offset, const unsigned n_offset) {
		return (mem_offset % frag_m + m_offset) + ((mem_offset / frag_m) + n_offset) * ldm;
	}
	// calculate memory ofset from matrix position (i,j) given by foreach_ij
	__device__ unsigned operator()(const unsigned i, const unsigned j, const unsigned ldm, const unsigned m_offset, const unsigned n_offset) {
		return (i + m_offset) + (j + n_offset) * ldm;
	}
};

template <class Use, class T>
struct sub_frag_t {
	using type = T;
};
template <>
struct sub_frag_t<nvcuda::wmma::accumulator, half                         > {using type = float;};
template <>
struct sub_frag_t<nvcuda::wmma::accumulator, nvcuda::wmma::precision::tf32> {using type = float;};

template <class Layout, int a, int b>
struct layout_switch;
template <int a, int b>
struct layout_switch<nvcuda::wmma::col_major, a, b> {const static int value = a;};
template <int a, int b>
struct layout_switch<nvcuda::wmma::row_major, a, b> {const static int value = b;};
} // detail
} // tcec
} // namespace wmma
} // namespace mtk
#endif

#ifndef __WMMAE_SIMT_DETAIL_M16N16K16__
#define __WMMAE_SIMT_DETAIL_M16N16K16__
#include <mma.h>
#include "common.hpp"
#include "fma.hpp"
#include "../../../../detail/common.hpp"

// Fragment layout
//                                   array_A/frag_A (len=8)
//                               |---------------------------|
//                               |  |  |  |  |         |  |  |
//                               |00|01|02|03|.........|14|15|
//                               |  |  |  |  |         |  |  |
//                               |  |  |  |  |         |  |  |
//                               |--|--|--|--|---------|--|--| 16
//                               |  |  |  |  |         |  |  |
//                               |16|17|18|19|.........|30|31|
//                               |  |  |  |  |         |  |  |
//                               |  |  |  |  |         |  |  |
//                               |---------------------------|
//              16
// |---------------------------| |---------------------------|
// |     00      |     16      | |  |  |  |  |         |  |  |
// |     01      |     17      | |00|01|02|03|.........|14|15|
// |     02      |     18      | |  |  |  |  |         |  |  |
// |     03      |     19      | |  |  |  |  |         |  |  |
// |      .      |      .      | |  |  |  |  |         |  |  | 16
// |      .      |      .      | |  |  |  |  |         |  |  |
// |      .      |      .      | |16|17|18|19|.........|30|31|
// |     14      |     30      | |  |  |  |  |         |  |  |
// |     15      |     31      | |  |  |  |  |         |  |  |
// |---------------------------| |---------------------------|
//     array_B/frag_B (len=8)           array_acc (len=16)
//
//                                             |
//                                             V
//
//                                     frag_C/D (len=8)
//                               |---------------------------|
//                               |  |  |  |  |         |  |  |
//                               |00|01|02|03|.........|14|15|
//                               |  |  |  |  |         |  |  |
//                               |  |  |  |  |         |  |  |
//                               |--|--|--|--|---------|--|--| 16
//                               |  |  |  |  |         |  |  |
//                               |16|17|18|19|.........|30|31|
//                               |  |  |  |  |         |  |  |
//                               |  |  |  |  |         |  |  |
//                               |---------------------------|
namespace mtk {
namespace wmma {
namespace mma_simt {
template <class T, class Layout>
class fragment<nvcuda::wmma::matrix_a   , 16, 16, 16, T, Layout> : public mtk::wmma::mma_simt::detail::__frag_base<T, 8>{};
template <class T, class Layout>
class fragment<nvcuda::wmma::matrix_b   , 16, 16, 16, T, Layout> : public mtk::wmma::mma_simt::detail::__frag_base<T, 8>{};
template <class T>
class fragment<nvcuda::wmma::accumulator, 16, 16, 16, T> : public mtk::wmma::mma_simt::detail::__frag_base<T, 8>{};

// foreach
template <class Func, class T>
__device__ void foreach(
		fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::col_major>& frag, const Func func
		) {
	const auto m = threadIdx.x & 0xf;
	const auto n_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m + 16 * (n_offset + i));}
	}
}

template <class Func, class T>
__device__ void foreach(
		fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major>& frag, const Func func
		) {
	const auto m = threadIdx.x & 0xf;
	const auto n_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m * 16 + (n_offset + i));}
	}
}

template <class Func, class T>
__device__ void foreach(
		fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major>& frag, const Func func
		) {
	const auto n = threadIdx.x & 0xf;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, n * 16 + (m_offset + i));}
	}
}

template <class Func, class T>
__device__ void foreach(
		fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major>& frag, const Func func
		) {
	const auto n = threadIdx.x & 0xf;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, n + 16 * (m_offset + i));}
	}
}

template <class Func, class T>
__device__ inline void foreach(mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout, const Func func) {
	const auto n = threadIdx.x & 0xf;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned i = 0; i < frag.num_elements; i++) {
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, n * 16 + (m_offset + i));}
		}
	} else {
		for (unsigned i = 0; i < frag.num_elements; i++) {
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, n + 16 * (m_offset + i));}
		}
	}
}

// foreach_ij
template <class Func, class T>
__device__ void foreach_ij(
		fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::col_major>& frag, const Func func
		) {
	const auto m = threadIdx.x & 0xf;
	const auto n_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m, n_offset + i);}
	}
}

template <class Func, class T>
__device__ void foreach_ij(
		fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major>& frag, const Func func
		) {
	const auto m = threadIdx.x & 0xf;
	const auto n_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m, n_offset + i);}
	}
}

template <class Func, class T>
__device__ void foreach_ij(
		fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major>& frag, const Func func
		) {
	const auto n = threadIdx.x & 0xf;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m_offset + i, n);}
	}
}

template <class Func, class T>
__device__ void foreach_ij(
		fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major>& frag, const Func func
		) {
	const auto n = threadIdx.x & 0xf;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m_offset + i, n);}
	}
}

template <class Func, class T>
__device__ inline void foreach_ij(mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout, const Func func) {
	const auto n = threadIdx.x & 0xf;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m_offset + i, n);}
	}
}

// foreach_v
template <class Func, class T>
__device__ void foreach_v(
		fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::col_major>& frag, const Func func
		) {
	const auto m = threadIdx.x & 0xf;
	if (threadIdx.x & 0b10000u) return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, m);}
}

template <class Func, class T>
__device__ void foreach_v(
		fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major>& frag, const Func func
		) {
	if (threadIdx.x & 0b01111u) return;
	const auto n_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, n_offset + i);}
	}
}

template <class Func, class T>
__device__ void foreach_v(
		fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major>& frag, const Func func
		) {
	if (threadIdx.x & 0b01111u) return;
	const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m_offset + i);}
	}
}

template <class Func, class T>
__device__ void foreach_v(
		fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major>& frag, const Func func
		) {
	const auto n = threadIdx.x & 0xf;
	if (threadIdx.x & 0b10000u) return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, n);}
}

template <class Func, class T>
__device__ inline void foreach_v(mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout, const Func func) {
	if (layout == nvcuda::wmma::mem_col_major) {
		if (threadIdx.x & 0b01111u) return;
		const auto m_offset = (mtk::wmma::detail::common::get_lane_id() >> 4) << 3;
		for (unsigned i = 0; i < frag.num_elements; i++) {
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, m_offset + i);}
		}
	} else {
		const auto n = threadIdx.x & 0xf;
		if (threadIdx.x & 0b10000u) return;

		{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, n);}
	}
}

// mma
template <class AB_T, class A_Layout, class B_Layout, class C_T, class D_T>
__device__ void mma_sync(
		fragment<nvcuda::wmma::accumulator, 16, 16, 16, D_T , void>& frag_d,
		const fragment<nvcuda::wmma::matrix_a   , 16, 16, 16, AB_T, A_Layout>& frag_a,
		const fragment<nvcuda::wmma::matrix_b   , 16, 16, 16, AB_T, B_Layout>& frag_b,
		const fragment<nvcuda::wmma::accumulator, 16, 16, 16, C_T , void>& frag_c,
		const mtk::wmma::mma_simt::detail::fma<AB_T, AB_T, C_T>& mfma = mtk::wmma::mma_simt::detail::fma<AB_T, AB_T, C_T>()
		) {
	AB_T array_a[frag_a.num_elements];
	AB_T array_b[frag_b.num_elements];
	C_T  array_acc[frag_c.num_elements * 2];

	// init A, B
	for (unsigned i = 0; i < frag_a.num_elements; i++) {
		array_a[i] = frag_a.x[i];
	}
	for (unsigned i = 0; i < frag_b.num_elements; i++) {
		array_b[i] = frag_b.x[i];
	}

	// matmul
	constexpr unsigned num_swaps = 15;
	constexpr unsigned swap_index_list[num_swaps] = {
		1, 2, 1, 4, 2, 1, 2, 8, 4, 2, 1, 2, 4, 2, 1
	};

	unsigned acc_index = threadIdx.x & 0xf;
	array_acc[acc_index] = detail::cast<C_T>(0);
	for (unsigned k = 0; k < frag_a.num_elements; k++) {
		array_acc[acc_index] = mfma(array_a[k], array_b[k], array_acc[acc_index]);
	}
	for (unsigned s = 0; s < num_swaps; s++) {
		const unsigned swap_index = swap_index_list[s];
		acc_index ^= swap_index;
		array_acc[acc_index] = detail::cast<C_T>(0);
		// swap a array
		for (unsigned k = 0; k < frag_a.num_elements; k++) {
			// swap
			array_a[k] = __shfl_xor_sync(0xffffffff, array_a[k], swap_index);
			// fma
			array_acc[acc_index] = mfma(array_a[k], array_b[k], array_acc[acc_index]);
		}
	}

	// collect C frag
	const auto offset_0 = (mtk::wmma::detail::common::get_lane_id() >> 4) * frag_c.num_elements;
	const auto offset_1 = frag_c.num_elements - offset_0;
	for (unsigned i = 0; i < frag_c.num_elements; i++) {
		frag_d.x[i] = array_acc[i + offset_0] + __shfl_xor_sync(0xffffffff, array_acc[i + offset_1], 16) + frag_c.x[i];
	}
}

} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif

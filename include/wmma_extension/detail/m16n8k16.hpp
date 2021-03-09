#ifndef __WMMAE_M16N8K16_HPP__
#define __WMMAE_M16N8K16_HPP__
#include <mma.h>
#include "common.hpp"

namespace mtk {
namespace wmma {

template <> class fragment<nvcuda::wmma::matrix_a   , 16, 8, 16, half , nvcuda::wmma::row_major> : public __frag_base<half, 8>{};
template <> class fragment<nvcuda::wmma::matrix_b   , 16, 8, 16, half , nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 16, 8, 16, float> : public __frag_base<float, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 16, 8, 16, half > : public __frag_base<half , 4>{};


// foreach
template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 16, 8, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned col_block_id = mtk::wmma::detail::common::get_lane_id() % 4;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	for (unsigned i = 0; i < 2; i++) {
		for (unsigned j = 0; j < 2; j++) {
			const auto col = i * 8 + col_block_id * 2;
			const auto row = row_block_id + j * 8;
			{const unsigned frag_index_list[1] = {(i * 4 + j * 2 + 0)};func(frag_index_list, 1, row + (col + 0) * 16);}
			{const unsigned frag_index_list[1] = {(i * 4 + j * 2 + 1)};func(frag_index_list, 1, row + (col + 1) * 16);}
		}
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 16, 8, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() % 4;

	for (unsigned i = 0; i < 2; i++) {
		const auto row = row_block_id * 2 + i * 8;
		{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, (row + 0) + col * 16);}
		{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, (row + 1) + col * 16);}
	}
}

template <class Func, class T>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 16, 8, T>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	for (unsigned i = 0; i < 2; i++) {
		const auto row = row_block_id2 + i * 8;
		if (layout == nvcuda::wmma::mem_col_major) {
			{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row + (col + 0) * 16);}
			{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row + (col + 1) * 16);}
		} else {
			{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row * 8 + (col + 0));}
			{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row * 8 + (col + 1));}
		}
	}
}
} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

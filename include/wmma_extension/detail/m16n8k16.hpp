#ifndef __WMMAE_M16N8K16_HPP__
#define __WMMAE_M16N8K16_HPP__
#include <mma.h>
#include "common.hpp"

namespace mtk {
namespace wmma {
namespace mma {
template <> class fragment<nvcuda::wmma::matrix_a   , 16, 8, 16, half , nvcuda::wmma::row_major> : public __frag_base<half, 8>{};
template <> class fragment<nvcuda::wmma::matrix_b   , 16, 8, 16, half , nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 16, 8, 16, float> : public __frag_base<float, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 16, 8, 16, half > : public __frag_base<half , 4>{};


// foreach
template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned col_block_id = mtk::wmma::detail::common::get_lane_id() % 4;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	for (unsigned i = 0; i < 2; i++) {
		for (unsigned j = 0; j < 2; j++) {
			const auto col = i * 8 + col_block_id * 2;
			const auto row = row_block_id + j * 8;
			{const unsigned frag_index_list[1] = {(i * 4 + j * 2 + 0)};func(frag_index_list, 1, row * 16 + (col + 0));}
			{const unsigned frag_index_list[1] = {(i * 4 + j * 2 + 1)};func(frag_index_list, 1, row * 16 + (col + 1));}
		}
	}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() % 4;

	for (unsigned i = 0; i < 2; i++) {
		const auto row = row_block_id * 2 + i * 8;
		{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, (row + 0) + col * 16);}
		{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, (row + 1) + col * 16);}
	}
}

template <class Func, class T>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16, T>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	for (unsigned i = 0; i < 2; i++) {
		const auto row = row_block_id + i * 8;
		if (layout == nvcuda::wmma::mem_col_major) {
			{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row + (col + 0) * 16);}
			{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row + (col + 1) * 16);}
		} else {
			{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row * 8 + (col + 0));}
			{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row * 8 + (col + 1));}
		}
	}
}

// foreach_ij
template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned col_block_id = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	for (unsigned i = 0; i < 2; i++) {
		for (unsigned j = 0; j < 2; j++) {
			const auto col = i * 8 + col_block_id;
			const auto row = row_block_id + j * 8;
			{const unsigned frag_index_list[1] = {(i * 4 + j * 2 + 0)};func(frag_index_list, 1, row, col + 0);}
			{const unsigned frag_index_list[1] = {(i * 4 + j * 2 + 1)};func(frag_index_list, 1, row, col + 1);}
		}
	}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
	const unsigned row_block_id = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;

	for (unsigned i = 0; i < 2; i++) {
		const auto row = row_block_id + i * 8;
		{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row + 0, col);}
		{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row + 1, col);}
	}
}

template <class Func, class T>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16, T>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	for (unsigned i = 0; i < 2; i++) {
		const auto row = row_block_id + i * 8;
		if (layout == nvcuda::wmma::mem_col_major) {
			{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row, col + 0);}
			{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row, col + 1);}
		} else {
			{const unsigned frag_index_list[1] = {(i * 2 + 0)};func(frag_index_list, 1, row, col + 0);}
			{const unsigned frag_index_list[1] = {(i * 2 + 1)};func(frag_index_list, 1, row, col + 1);}
		}
	}
}

// foreach_v
template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	if (mtk::wmma::detail::common::get_lane_id() >= 4)
		return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 1);}
	{const unsigned frag_index_list[1] = {4};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 8);}
	{const unsigned frag_index_list[1] = {5};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 9);}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	if (mtk::wmma::detail::common::get_lane_id() >= 4)
		return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 8);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 9);}
}

template <class Func, class T>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16, T>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	if (layout == nvcuda::wmma::mem_col_major) {
		if (mtk::wmma::detail::common::get_lane_id() & 0b11)
			return;
		{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() / 4 + 0);}
		{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() / 4 + 8);}
	} else {
		if (mtk::wmma::detail::common::get_lane_id() >= 4)
			return;
		{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 0);}
		{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() * 2 + 1);}
	}
}

// Mma
__device__ inline void mma_sync(
		fragment<nvcuda::wmma::accumulator, 16, 8, 16, float>& d,
		const fragment<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>& a,
		const fragment<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>& b,
		const fragment<nvcuda::wmma::accumulator, 16, 8, 16, float>& c) {
	asm(R"({
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
      {%0, %1, %2, %3},
      {%4, %5, %6, %7},
      {%8, %9},
      {%10, %11, %12, %13};
})"
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
			: "r"(*reinterpret_cast<const unsigned*>(a.x)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 2)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 4)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 6)),
			"r"(*reinterpret_cast<const unsigned*>(b.x)),
			"r"(*reinterpret_cast<const unsigned*>(b.x + 2)),
			"f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

__device__ inline void mma_sync(
		fragment<nvcuda::wmma::accumulator, 16, 8, 16, half>& d,
		const fragment<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>& a,
		const fragment<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>& b,
		const fragment<nvcuda::wmma::accumulator, 16, 8, 16, half>& c) {
	asm(R"({
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
      {%0, %1},
      {%2, %3, %4, %5},
      {%6, %7},
      {%8, %9};
})"
			: "=r"(*reinterpret_cast<unsigned*>(d.x)),
			"=r"(*reinterpret_cast<unsigned*>(d.x + 2))
			: "r"(*reinterpret_cast<const unsigned*>(a.x)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 2)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 4)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 6)),
			"r"(*reinterpret_cast<const unsigned*>(b.x)),
			"r"(*reinterpret_cast<const unsigned*>(b.x + 2)),
			"r"(*reinterpret_cast<const unsigned*>(c.x)),
			"r"(*reinterpret_cast<const unsigned*>(c.x + 2)));
}
} // namespace mma
} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

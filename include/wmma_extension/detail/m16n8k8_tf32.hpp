#ifndef __WMMAE_M16N8K8_TF32_HPP__
#define __WMMAE_M16N8K8_TF32_HPP__
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
#include <mma.h>
#include "common.hpp"

namespace mtk {
namespace wmma {
namespace mma {
template <> class fragment<nvcuda::wmma::matrix_a   , 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> : public __frag_base<float, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b   , 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> : public __frag_base<float, 2>{};
// The accumulator is same with m16n8k8-float for nvcuda::wmma::precision::tf32
//template <> class fragment<nvcuda::wmma::accumulator, 16, 8, 8, float> : public __frag_base<float, 4>{};

// foreach
template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4);
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, (row_block_id + 0) * 8 + (col + 0));}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, (row_block_id + 8) * 8 + (col + 0));}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, (row_block_id + 0) * 8 + (col + 4));}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, (row_block_id + 8) * 8 + (col + 4));}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
	const unsigned row_start = mtk::wmma::detail::common::get_lane_id() % 4;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, (row_start + 0) + col * 8);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, (row_start + 4) + col * 8);}
}

// foreach_ij
template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4);
	const unsigned row_block_id = mtk::wmma::detail::common::get_lane_id() / 4;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, (row_block_id + 0), (col + 0));}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, (row_block_id + 8), (col + 0));}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, (row_block_id + 0), (col + 4));}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, (row_block_id + 8), (col + 4));}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
	const unsigned row_start = mtk::wmma::detail::common::get_lane_id() % 4;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, (row_start + 0), col);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, (row_start + 4), col);}
}

// foreach_v
template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	if (mtk::wmma::detail::common::get_lane_id() >= 4)
		return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, (mtk::wmma::detail::common::get_lane_id() + 0));}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, (mtk::wmma::detail::common::get_lane_id() + 4));}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	if (mtk::wmma::detail::common::get_lane_id() >= 4)
		return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mtk::wmma::detail::common::get_lane_id() + 4);}
}

// Mma
__device__ inline void mma_sync(
		fragment<nvcuda::wmma::accumulator, 16, 8, 8, float>& d,
		const fragment<nvcuda::wmma::matrix_a, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& a,
		const fragment<nvcuda::wmma::matrix_b, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& b,
		const fragment<nvcuda::wmma::accumulator, 16, 8, 8, float>& c) {
	asm(R"({
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
      {%0, %1, %2, %3},
      {%4, %5, %6, %7},
      {%8, %9},
      {%10, %11, %12, %13};
})"
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
			: "r"(*reinterpret_cast<const unsigned*>(a.x)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 1)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 2)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 3)),
			"r"(*reinterpret_cast<const unsigned*>(b.x)),
			"r"(*reinterpret_cast<const unsigned*>(b.x + 1)),
			"f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}
} // namespace mma
} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

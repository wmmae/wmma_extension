#ifndef __WMMAE_DETAIL_80_TF32_HPP__
#define __WMMAE_DETAIL_80_TF32_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_80 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 4) + (lane_id >> 2);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x2) << 5) + ((x & 0x1) << 3);

		const unsigned frag_index_list[1] = {x};func(frag_index_list, 1, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id & 0x3) + ((lane_id & 0x1c) << 1);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 6) + ((x & 0x2) << 1);

		const unsigned frag_index_list[1] = {x};func(frag_index_list, 1, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id & 0x3) + ((lane_id & 0x1c) << 1);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 2) + ((x & 0x2) << 5);

		const unsigned frag_index_list[1] = {x};func(frag_index_list, 1, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 4) + (lane_id >> 2);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 6) + ((x & 0x2) << 2);

		const unsigned frag_index_list[1] = {x};func(frag_index_list, 1, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		{
			const unsigned frag_index_list[1] = {0};
			func(frag_index_list, 1, index_offset);
		}
		{
			const unsigned frag_index_list[1] = {1};
			func(frag_index_list, 1, index_offset + 8);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		{
			const unsigned frag_index_list[1] = {0};
			func(frag_index_list, 1, index_offset);
		}
		{
			const unsigned frag_index_list[1] = {2};
			func(frag_index_list, 1, index_offset + 4);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		{
			const unsigned frag_index_list[1] = {0};
			func(frag_index_list, 1, index_offset);
		}
		{
			const unsigned frag_index_list[1] = {1};
			func(frag_index_list, 1, index_offset + 4);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		{
			const unsigned frag_index_list[1] = {0};
			func(frag_index_list, 1, index_offset);
		}
		{
			const unsigned frag_index_list[1] = {2};
			func(frag_index_list, 1, index_offset + 8);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const bool load_flag = (lane_id & 0x3) == 0;
		const unsigned index_offset = lane_id >> 2;
		if (load_flag) {
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, index_offset + 0);}
			{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, index_offset + 8);}
		}
	} else {
		bool load_flag = (lane_id & 0b11100) == 0;
		const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
		if (load_flag) {
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, index_offset + 0);}
			{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, index_offset + 1);}
			{const unsigned frag_index_list[1] = {4};func(frag_index_list, 1, index_offset + 8);}
			{const unsigned frag_index_list[1] = {5};func(frag_index_list, 1, index_offset + 9);}
		}
	}
}
template <class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, T>& frag, const T alpha) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto lane_mod_9 = lane_id % 9;
	const bool set_flag = (lane_mod_9 == 0) || (lane_mod_9 == 4);
	if (set_flag) {
		frag.x[(lane_mod_9 >> 2) + 0] += alpha;
		frag.x[(lane_mod_9 >> 2) + 6] += alpha;
	}
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag_a,
		const T* const a, const S* const da,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_a);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (CORRECTION_TERMS == 2 && ((lane_id & 0x3) == 0x3)) {
		return;
	}

	const auto a_ptr = (lane_id & 0x1) ? da : a;

	frag_a.x[0] = a_ptr[(lane_id >> 2) + 0];
	frag_a.x[1] = a_ptr[(lane_id >> 2) + 8];
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag_b,
		const T* const b, const S* const db,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_b);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (CORRECTION_TERMS == 2 && ((lane_id & 0x3) == 0x3)) {
		return;
	}

	const auto b_ptr = (lane_id & 0x2) ? db : b;

	frag_b.x[0] = b_ptr[(lane_id >> 2) + 0];
	frag_b.x[2] = b_ptr[(lane_id >> 2) + 8];
}

template <unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag_a,
		const float* const a_ptr,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_a);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (CORRECTION_TERMS == 2 && ((lane_id & 0x3) == 0x3)) {
		return;
	}

	auto a0 = a_ptr[(lane_id >> 2) + 0];
	auto a8 = a_ptr[(lane_id >> 2) + 8];

	// __float2tf32
#if __CUDA_ARCH__ >= 800
	if (lane_id & 0x1) {
		a0 = a0 - mtk::wmma::detail::common::to_tf32(a0);
		a8 = a8 - mtk::wmma::detail::common::to_tf32(a8);
	}
#endif

	frag_a.x[0] = mtk::wmma::detail::common::to_tf32(a0);
	frag_a.x[1] = mtk::wmma::detail::common::to_tf32(a8);
}

template <unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag_b,
		const float* const b_ptr,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_b);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (CORRECTION_TERMS == 2 && ((lane_id & 0x3) == 0x3)) {
		return;
	}

	auto b0 = b_ptr[(lane_id >> 2) + 0];
	auto b8 = b_ptr[(lane_id >> 2) + 8];

	// __float2tf32
#if __CUDA_ARCH__ >= 800
	if (lane_id & 0x2) {
		b0 = b0 - mtk::wmma::detail::common::to_tf32(b0);
		b8 = b8 - mtk::wmma::detail::common::to_tf32(b8);
	}
#endif

	frag_b.x[0] = mtk::wmma::detail::common::to_tf32(b0);
	frag_b.x[2] = mtk::wmma::detail::common::to_tf32(b8);
}
#endif /* __CUDA_ARCH__ >= 8000 */
} // namespace sm_80
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

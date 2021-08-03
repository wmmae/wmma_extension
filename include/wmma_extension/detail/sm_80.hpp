#ifndef __WMMAE_DETAIL_80_HPP__
#define __WMMAE_DETAIL_80_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_80 {
template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned offset = ((x & 0x1) << 4) + ((x & 0x2) << 2) + ((x & 0x4) << 5);
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned offset = (x & 0x1) + ((x & 0x2) << 6) + ((x & 0x4) << 1);
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned offset = (x & 0x1) + ((x & 0x2) << 2) + ((x & 0x4) << 5);
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned offset = ((x & 0x1) << 4) + ((x & 0x2) << 6) + ((x & 0x4) << 1);
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, start_index + offset);
	}
}

template <class T, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const unsigned start_index = (lane_id & 0b11) * 32 + (lane_id >> 2);
		for (unsigned x = 0; x < frag.num_elements; x++) {
			const unsigned index = start_index + (x & 0b1) * 16 + (x & 0b10) * 4 + (x & 0b100) * 32;
			const unsigned frag_index_list[1] = {x};
			func(frag_index_list, 1, index);
		}
	} else {
		const unsigned start_index = (lane_id & 0b11) * 2 + (lane_id & 0b11100) * 4;
		for (unsigned x = 0; x < frag.num_elements; x++) {
			const unsigned index = start_index + (x & 0b1) + (x & 0b10) * 64 + (x & 0b100) * 2;
			const unsigned frag_index_list[1] = {x};
			func(frag_index_list, 1, index);
		}
	}
}

// ---------------------------------
// foreach_ij
// ---------------------------------
template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = lane_id / 4;
	const auto j_offset = (lane_id & 0b11) * 2;
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned i = i_offset + (x & 0b10) * 4;
		const unsigned j = j_offset + (x & 0b1) + (x & 0b100) * 2;
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, i, j);
	}
}

template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = (lane_id & 0b11) * 2;
	const auto j_offset = lane_id / 4;
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned i = i_offset + (x & 0b1) + (x & 0b10) * 4;
		const unsigned j = j_offset + (x & 0b100) * 2;
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, i, j);
	}
}

template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = lane_id / 4;
	const auto j_offset = (lane_id & 0b11) * 2;
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned i = i_offset + (x & 0b10) * 4;
		const unsigned j = j_offset + (x & 0b1) + (x & 0b100) * 2;
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, i, j);
	}
}

template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = (lane_id & 0b11) * 2;
	const auto j_offset = lane_id / 4;
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned i = i_offset + (x & 0b1) + (x & 0b10) * 4;
		const unsigned j = j_offset + (x & 0b100) * 2;
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, i, j);
	}
}

template <class T, class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row_start = (lane_id >> 2);
	const unsigned col_start = (lane_id & 0b11) * 2;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned col = col_start + (x & 0b100) * 2 + (x & 0b1);
		const unsigned row = row_start + (x & 0b10) * 4;
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, row, col);
	}
}

// ---------------------------------
// foreach_v
// ---------------------------------
template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		{
			const unsigned frag_index_list[2] = {0, 8};
			func(frag_index_list, 2, index_offset);
		}
		{
			const unsigned frag_index_list[2] = {2, 10};
			func(frag_index_list, 2, index_offset + 8);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		{
			const unsigned frag_index_list[2] = {0, 8};
			func(frag_index_list, 2, index_offset);
		}
		{
			const unsigned frag_index_list[2] = {1, 9};
			func(frag_index_list, 2, index_offset + 1);
		}
		{
			const unsigned frag_index_list[2] = {4, 12};
			func(frag_index_list, 2, index_offset + 8);
		}
		{
			const unsigned frag_index_list[2] = {5, 13};
			func(frag_index_list, 2, index_offset + 9);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		{
			const unsigned frag_index_list[2] = {0, 8};
			func(frag_index_list, 2, index_offset);
		}
		{
			const unsigned frag_index_list[2] = {1, 9};
			func(frag_index_list, 2, index_offset + 1);
		}
		{
			const unsigned frag_index_list[2] = {2, 10};
			func(frag_index_list, 2, index_offset + 8);
		}
		{
			const unsigned frag_index_list[2] = {3, 11};
			func(frag_index_list, 2, index_offset + 9);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	bool load_flag = (lane_id & 0x3) == 0;
	unsigned long index_offset = lane_id >> 2;
	if(load_flag) {
		{
			const unsigned frag_index_list[2] = {0, 8};
			func(frag_index_list, 2, index_offset);
		}
		{
			const unsigned frag_index_list[2] = {4, 12};
			func(frag_index_list, 2, index_offset + 8);
		}
	}
}

template <class Func, class T>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout, Func func) {
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

// map function
__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b111) * 4 + (j & 0b111) / 2;
	const auto fid_head = (i & 0b1000) / 4 + (j & 0b1) + (j & 0b1000) / 2;

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head;
	fid_list[1] = fid_head + 8;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b111) / 2 + (j & 0b111) * 4;
	const auto fid_head = (i & 0b1) + (i & 0b1000) / 4 + (j & 0b1000) / 2;

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head;
	fid_list[1] = fid_head + 8;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b111) * 4 + (j & 0b111) / 2;
	const auto fid_head = (i & 0b1000) / 4 + (j & 0b1) + (j & 0b1000) / 2;

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head;
	fid_list[1] = fid_head + 8;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b111) / 2 + (j & 0b111) * 4;
	const auto fid_head = (i & 0b1) + (i & 0b1000) / 4 + (j & 0b1000) / 2;

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head;
	fid_list[1] = fid_head + 8;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half, void>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b111) * 4 + (j & 0b111) / 2;
	const auto fid_head = (i & 0b1000) / 4 + (j & 0b1) + (j & 0b1000) / 2;

	list_size = 1;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, void>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b111) * 4 + (j & 0b111) / 2;
	const auto fid_head = (i & 0b1000) / 4 + (j & 0b1) + (j & 0b1000) / 2;

	list_size = 1;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
}

template <class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
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
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag_a,
		const T* const a, const S* const da,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_a);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (lane_id & 0x2) {
		return;
	}

	frag_a.x[0] = frag_a.x[8 + 0] = common::cast<half>(a[(lane_id >> 2) + 0]);
	frag_a.x[2] = frag_a.x[8 + 2] = common::cast<half>(a[(lane_id >> 2) + 8]);

	if (CORRECTION_TERMS == 3 || ((lane_id & 0x1) == 0)) {
		frag_a.x[1] = frag_a.x[8 + 1] = common::cast<half>(da[(lane_id >> 2) + 0]);
		frag_a.x[3] = frag_a.x[8 + 3] = common::cast<half>(da[(lane_id >> 2) + 8]);
	}
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag_b,
		const T* const b, const S* const db,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_b);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (lane_id & 0x2) {
		return;
	}

	const auto bp = (lane_id & 0x1) ? db : b;

	const auto b0 = bp[(lane_id >> 2) + 0];
	const auto b8 = bp[(lane_id >> 2) + 8];

	frag_b.x[0] = frag_b.x[8 + 0] = common::cast<half>(b0);
	frag_b.x[4] = frag_b.x[8 + 4] = common::cast<half>(b8);

	if (CORRECTION_TERMS == 3 || ((lane_id & 0x3) == 0)) {
		frag_b.x[1] = frag_b.x[8 + 1] = frag_b.x[0];
		frag_b.x[5] = frag_b.x[8 + 5] = frag_b.x[4];
	}
}

template <unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag_a,
		const float* const a_ptr,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_a);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (lane_id & 0x2) {
		return;
	}

	const auto a0 = a_ptr[(lane_id >> 2) + 0];
	const auto a8 = a_ptr[(lane_id >> 2) + 8];

	frag_a.x[0] = frag_a.x[8 + 0] = common::cast<half>(a0);
	frag_a.x[2] = frag_a.x[8 + 2] = common::cast<half>(a8);

	if ((CORRECTION_TERMS == 3) || ((lane_id & 0x1) == 0)) {
		frag_a.x[1] = frag_a.x[8 + 1] = common::cast<half>(a0 - common::cast<float>(frag_a.x[0]));
		frag_a.x[3] = frag_a.x[8 + 3] = common::cast<half>(a8 - common::cast<float>(frag_a.x[2]));
	}
}

template <unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag_b,
		const float* const b_ptr,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_b);
	}
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (lane_id & 0x2) {
		return;
	}

	auto b0 = b_ptr[(lane_id >> 2) + 0];
	auto b8 = b_ptr[(lane_id >> 2) + 8];

	if (lane_id & 0x1) {
		b0 = b0 - common::cast<float>(common::cast<half>(b0));
		b8 = b8 - common::cast<float>(common::cast<half>(b8));
	}

	frag_b.x[0] = frag_b.x[8 + 0] = common::cast<half>(b0);
	frag_b.x[4] = frag_b.x[8 + 4] = common::cast<half>(b8);

	if (CORRECTION_TERMS == 3 || ((lane_id & 0x3) == 0)) {
		frag_b.x[1] = frag_b.x[8 + 1] = frag_b.x[0];
		frag_b.x[5] = frag_b.x[8 + 5] = frag_b.x[4];
	}
}
} // namespace sm_80
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

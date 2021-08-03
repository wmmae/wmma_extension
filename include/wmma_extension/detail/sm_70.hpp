#ifndef __WMMAE_DETAIL_70_HPP__
#define __WMMAE_DETAIL_70_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_70 {

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (((lane_id >> 2) & 0b1) << 3) + ((lane_id >> 4) << 2) + ((lane_id & 0b11) << 4);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;

		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (((lane_id >> 2) & 0b1) << 7) + ((lane_id >> 4) << 6) + ((lane_id & 0b11) << 4);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = x;
		const unsigned index = start_index + offset;

		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (((lane_id >> 3) & 0b1) << 7) + ((lane_id >> 4) << 6) + ((lane_id & 0b11) << 4);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = x;
		const unsigned index = start_index + offset;

		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (((lane_id >> 3) & 0b1) << 3) + ((lane_id >> 4) << 2) + ((lane_id & 0b11) << 4);
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;

		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, index);
	}
}

template <class T, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_row_major) {
		const unsigned start_index = (lane_id & 0b11) * 16 + ((lane_id >> 2) & 0b1) * 128 + ((lane_id >> 3) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 8;
		for (unsigned x = 0; x < frag.num_elements; x++) {
			const unsigned frag_index_list[1] = {x};
			func(frag_index_list, 1, start_index + x);
		}
	} else {
		const unsigned start_index = (lane_id & 0b11) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 3) & 0b1) * 128 + ((lane_id >> 4) & 0b1) * 4;
		for (unsigned x = 0; x < frag.num_elements; x++) {
			const unsigned frag_index_list[1] = {x};
			func(frag_index_list, 1, start_index + x * 16);
		}
	}
}

template <class T, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_row_major) {
		const unsigned start_index = (lane_id & 0b1) + ((lane_id >> 1) & 0b1) * 32 + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 3) & 0b1) * 128 + ((lane_id >> 4) & 0b1) * 4;
		for (unsigned x = 0; x < frag.num_elements; x++) {
			const unsigned frag_index_list[1] = {x};
			func(frag_index_list, 1, start_index + (x & 0b1) * 16 + ((x >> 1) & 0b1) * 2 + (x >> 2) * 64);
		}
	} else {
		const unsigned start_index = (lane_id & 0b1) + (lane_id & 0b10) * 16 + (lane_id & 0b100) * 2 + (lane_id & 0b1000) * 16 + ((lane_id & 0b10000) >> 2);
		for (unsigned x = 0; x < frag.num_elements; x++) {
			const unsigned frag_index_list[1] = {x};
			func(frag_index_list, 1, start_index + (x & 0b1) * 16 + (x & 0b10) + (x & 0b100) * 16);
		}
	}
}

// -------------------------------
// foreach_ij
// -------------------------------
template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = (lane_id & 0b100) * 2 + (lane_id & 0b10000) / 4;
	const auto j_offset = lane_id & 0b11;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned i = i_offset + (x & 0b11);
		const unsigned j = j_offset + (x & 0b1100);
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, i, j);
	}
}

template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = 0;
	const auto j_offset = (lane_id & 0b11) + (lane_id & 0b1000) + (lane_id & 0b10000) / 4;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned i = i_offset + x;
		const unsigned j = j_offset;
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, i, j);
	}
}

template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = (lane_id & 0b11) + (lane_id & 0b100) * 2 + (lane_id & 0b10000) / 4;
	const auto j_offset = 0;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned i = i_offset;
		const unsigned j = j_offset + x;
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, i, j);
	}

}

template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = lane_id & 0b11;
	const auto j_offset = (lane_id & 0b1000) + (lane_id & 0b10000) / 4;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned i = i_offset + (x & 0b1100);
		const unsigned j = j_offset + (x & 0b11);
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, i, j);
	}
}

template <class T, class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0b11) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const auto col = x + ((lane_id >> 3) & 0b1) * 8;
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, row, col);
	}
}

template <class T, class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row_start = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4;
	const unsigned col_start = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const auto col = col_start + (x & 0b1) + ((x >> 2) & 0b1) * 4;
		const auto row = row_start + ((x >> 1) & 0b1) * 2;
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, row, col);
	}
}

// -------------------------------
// foreach_v
// -------------------------------
template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = ((lane_id >> 4) << 2) + (((lane_id >> 2) & 0x1) << 3);
	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			const unsigned frag_index_list[1] = {i};
			func(frag_index_list, 1, i + index_offset);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id == 0) || (lane_id == 8);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			const unsigned frag_index_list[1] = {i};
			func(frag_index_list, 1, i);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const bool load_flag = (lane_id == 0) || (lane_id == 4);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			const unsigned frag_index_list[1] = {i};
			func(frag_index_list, 1, i);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = ((lane_id >> 4) << 2) + (((lane_id >> 3) & 0x1) << 3);
	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			const unsigned frag_index_list[1] = {i};
			func(frag_index_list, 1, i + index_offset);
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if (!(tid & 0b01000)) {
			const auto mem_index = ((tid & 0b10000) >> 2) + (tid & 0x3) + ((tid & 0x4) << 1);
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_index + 0);}
		}
	} else {
		if (tid == 0 || tid == 8) {
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, tid + 0);}
			{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, tid + 1);}
			{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, tid + 2);}
			{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, tid + 3);}
			{const unsigned frag_index_list[1] = {4};func(frag_index_list, 1, tid + 4);}
			{const unsigned frag_index_list[1] = {5};func(frag_index_list, 1, tid + 5);}
			{const unsigned frag_index_list[1] = {6};func(frag_index_list, 1, tid + 6);}
			{const unsigned frag_index_list[1] = {7};func(frag_index_list, 1, tid + 7);}
		}
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if (!(tid & 0b10) && !(tid & 0b1000)) {
			const auto mem_index = ((tid & 0b10000) >> 2) + (tid & 0b1) + ((tid & 0b100) << 1);
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_index + 0);}
			{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_index + 2);}
		}
	} else {
		if (!(tid & 0b1) && !(tid & 0b10000) && !(tid & 0b100)) {
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, tid + 0);}
			{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, tid + 1);}
			{const unsigned frag_index_list[1] = {4};func(frag_index_list, 1, tid + 4);}
			{const unsigned frag_index_list[1] = {5};func(frag_index_list, 1, tid + 5);}
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
	const auto tid_head = (j & 0b11) + (i & 0b100) * 4 + (i & 0b1000) / 2;
	const auto fid_head = (i & 0b11) + (j & 0b1100);

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head + 8;
	fid_list[1] = fid_head;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (j & 0b11) + (j & 0b100) * 4 + (j & 0b1000);
	const auto fid_head = i;

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head + 4;
	fid_list[1] = fid_head;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b100) * 4 + (i & 0b1000) / 2 + (i & 0b11);
	const auto fid_head = j;

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head + 8;
	fid_list[1] = fid_head;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b11) + (j & 0b100) * 4 + (j & 0b1000);
	const auto fid_head = (i & 0b1100) + (j & 0b11);

	list_size = 2;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
	tid_list[1] = tid_head + 4;
	fid_list[1] = fid_head;
}

__device__ inline void map(
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half, void>& frag,
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	const auto tid_head = (i & 0b11) + (i & 0b100) * 4 + (i & 0b1000) / 2 + (j & 0b1000);
	const auto fid_head = j & 0b111;

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
	const auto tid_head = (i & 0b1) + (i & 0b100) * 4 + (i & 0b1000) / 2 + (j & 0b1010);
	const auto fid_head = (i & 0b10) + (j & 0b101);

	list_size = 1;
	tid_list[0] = tid_head;
	fid_list[0] = fid_head;
}

template <class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (std::is_same<T, float>::value) {
		int4* const i4 = reinterpret_cast<int4*>(frag.x);
		i4[0] = make_int4(0, 0, 0, 0);
		i4[1] = make_int4(0, 0, 0, 0);
	} else {
		int4* const i4 = reinterpret_cast<int4*>(frag.x);
		i4[0] = make_int4(0, 0, 0, 0);
	}

	unsigned index_offset = 0;
	if(lane_id >> 4) {
		index_offset = 4;
	}

	const unsigned p0 = (lane_id >> 2) & 0x3;
	if(p0 == 0 || p0 == 3) {
		frag.x[(lane_id & 0x3) + index_offset] = common::cast<T>(1.0f);
	}
}

template <class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	unsigned index_offset = 0;
	if(lane_id >> 4) {
		index_offset = 4;
	}

	const unsigned p0 = (lane_id >> 2) & 0x3;
	if(p0 == 0 || p0 == 3) {
		frag.x[(lane_id & 0x3) + index_offset] += alpha;
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

	if (CORRECTION_TERMS == 2 && (lane_id & 0x3) == 0x3) {
		return;
	}

	const T* const a_ptr = ((lane_id & 0x1) == 0) ? a : da;
	const unsigned a_offset = ((lane_id & 0x10) >> 2) + ((lane_id & 0x4) << 1);

	frag_a.x[0] = detail::common::cast<half>(a_ptr[a_offset + 0]);
	frag_a.x[1] = detail::common::cast<half>(a_ptr[a_offset + 1]);
	frag_a.x[2] = detail::common::cast<half>(a_ptr[a_offset + 2]);
	frag_a.x[3] = detail::common::cast<half>(a_ptr[a_offset + 3]);
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

	if (CORRECTION_TERMS == 2 && (lane_id & 0x3) == 0x3) {
		return;
	}

	const T* const b_ptr = ((lane_id & 0x2) == 0) ? b : db;
	const unsigned b_offset = ((lane_id & 0x10) >> 2) + (lane_id & 0x8);

	frag_b.x[0] = detail::common::cast<half>(b_ptr[b_offset + 0]);
	frag_b.x[1] = detail::common::cast<half>(b_ptr[b_offset + 1]);
	frag_b.x[2] = detail::common::cast<half>(b_ptr[b_offset + 2]);
	frag_b.x[3] = detail::common::cast<half>(b_ptr[b_offset + 3]);
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

	if (CORRECTION_TERMS == 2 && (lane_id & 0x3) == 0x3) {
		return;
	}

	const bool is_residual = ((lane_id & 0x1) != 0);
	const unsigned a_offset = ((lane_id & 0x10) >> 2) + ((lane_id & 0x4) << 1);

#pragma unroll
	for (unsigned i = 0; i < 4; i++) {
		const auto a_fp32 = a_ptr[a_offset + i];
		frag_a.x[i] = detail::common::cast<half>(a_fp32);
		if (is_residual)
			frag_a.x[i] = detail::common::cast<half>(a_fp32 - detail::common::cast<float>(frag_a.x[i]));
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

	if (CORRECTION_TERMS == 2 && (lane_id & 0x3) == 0x3) {
		return;
	}

	const bool is_residual = ((lane_id & 0x2) != 0);
	const unsigned b_offset = ((lane_id & 0x10) >> 2) + (lane_id & 0x8);
#pragma unroll
	for (unsigned i = 0; i < 4; i++) {
		const auto b_fp32 = b_ptr[b_offset + i];
		frag_b.x[i] = detail::common::cast<half>(b_fp32);
		if (is_residual)
			frag_b.x[i] = detail::common::cast<half>(b_fp32 - detail::common::cast<float>(frag_b.x[i]));
	}
}
} // namespace sm_70
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

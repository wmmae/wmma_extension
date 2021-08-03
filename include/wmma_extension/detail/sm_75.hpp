#ifndef __WMMAE_DETAIL_75_HPP__
#define __WMMAE_DETAIL_75_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_75 {
template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id >> 2) + ((lane_id & 0b11) << 5);
	for (unsigned i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);

		const unsigned frag_index_list[2] = {i, j};
		func(frag_index_list, 2, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id >> 2) << 4) + ((lane_id & 0b11) << 1);
	for (unsigned i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);

		const unsigned frag_index_list[2] = {i, j};
		func(frag_index_list, 2, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id >> 2) << 4) + ((lane_id & 0b11) << 1);
	for (unsigned i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);

		const unsigned frag_index_list[2] = {i, j};
		func(frag_index_list, 2, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id >> 2) + ((lane_id & 0b11) << 5);
	for (unsigned i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);

		const unsigned frag_index_list[2] = {i, j};
		func(frag_index_list, 2, index);
	}
}

template <class T, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const unsigned start_index = (lane_id & 0b11) * 32 + (lane_id >> 2);
		for (unsigned i = 0; i < frag.num_elements; i++) {
			const unsigned index = start_index + (i & 0b1) * 16 + ((i >> 1) & 0b1) * 8 + ((i >> 2) & 0b1) * 128;
			const unsigned frag_index_list[1] = {i};
			func(frag_index_list, 1, index);
		}
	} else {
		const unsigned start_index = (lane_id & 0b11) * 2 + ((lane_id >> 2) & 1) * 16 + ((lane_id >> 3) & 0b1) * 32 + ((lane_id >> 4) & 0b1) * 64;
		for (unsigned i = 0; i < frag.num_elements; i++) {
			const unsigned index = start_index + (i & 0b1) + ((i >> 1) & 0b1) * 128 + ((i >> 2) & 0b1) * 8;
			const unsigned frag_index_list[1] = {i};
			func(frag_index_list, 1, index);
		}
	}
}

// ----------------------------------
// foreach_ij
// ----------------------------------
template <class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const auto i_offset = lane_id / 4;
	const auto j_offset = (lane_id & 0b11) * 2;
	for (unsigned x = 0; x < frag.num_elements / 2; x++) {
		const unsigned i = i_offset + (x & 0b10) * 4;
		const unsigned j = j_offset + (x & 0b100) * 2 + (x & 0b1);
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
		const unsigned i = i_offset + (x & 0b10) * 4 + (x & 0b1);
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
		const unsigned j = j_offset + (x & 0b100) * 2 + (x & 0b1);
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
		const unsigned i = i_offset + (x & 0b10) * 4 + (x & 0b1);
		const unsigned j = j_offset + (x & 0b100) * 2;
		const unsigned frag_index_list[2] = {x, x + 8};
		func(frag_index_list, 2, i, j);
	}
}

template <class T, class Func>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T, void>& frag, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row_start = lane_id >> 2;
	for (unsigned x = 0; x < frag.num_elements; x++) {
		const unsigned col = (lane_id & 0b11) * 2 + (x & 0b1) + (x >> 2) * 8;
		const unsigned row = row_start + (x & 0b10) * 4;
		const unsigned frag_index_list[1] = {x};
		func(frag_index_list, 1, row, col);
	}
}

// ----------------------------------
// foreach_v
// ----------------------------------
template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id >> 2;
	const bool load_flag = (lane_id & 0x3) == 0;
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
	const unsigned long index_offset = lane_id * 2;
	const bool load_flag = lane_id < 4;
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
	const unsigned long index_offset = lane_id * 2;
	const bool load_flag = lane_id < 4;
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
	const unsigned long index_offset = lane_id >> 2;

	const bool load_flag = (lane_id & 0x3) == 0;
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
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if ((tid & 0x3) == 0) {
			const auto mem_index = tid >> 2;
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_index + 0);}
			{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_index + 8);}
		}
	} else {
		if (!(tid & 0b11100)) {
			const auto mem_index = tid << 1;
			{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_index + 0);}
			{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_index + 1);}
			{const unsigned frag_index_list[1] = {4};func(frag_index_list, 1, mem_index + 8);}
			{const unsigned frag_index_list[1] = {5};func(frag_index_list, 1, mem_index + 9);}
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

	const unsigned mod9 = lane_id % 9;

	unsigned index_offset = mod9 >> 2;
	bool set_flag = mod9 == 0 || mod9 == 4;

	if(set_flag) {
		frag.x[index_offset] = frag.x[index_offset + 6] = common::cast<T>(1.0f);
	}
}

template <class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned mod9 = lane_id % 9;

	unsigned index_offset = mod9 >> 2;
	bool set_flag = mod9 == 0 || mod9 == 4;

	if(set_flag) {
		frag.x[index_offset] += alpha;
		frag.x[index_offset + 6] += alpha;
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

	if (lane_id & 0x2) return;

	// load a
	const unsigned offset = (lane_id >> 2);

	frag_a.x[ 0] = detail::common::cast<half>(a[offset + 0]);
	frag_a.x[ 2] = detail::common::cast<half>(a[offset + 8]);
	frag_a.x[ 8] = frag_a.x[ 0];
	frag_a.x[10] = frag_a.x[ 2];
	if (CORRECTION_TERMS == 3 || (lane_id & 0x1) == 0) {
		frag_a.x[ 0 + 1] = detail::common::cast<half>(da[offset + 0]);
		frag_a.x[ 2 + 1] = detail::common::cast<half>(da[offset + 8]);
		frag_a.x[ 8 + 1] = frag_a.x[ 0 + 1];
		frag_a.x[10 + 1] = frag_a.x[ 2 + 1];
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

	if (lane_id & 0x2) return;

	// load a
	const unsigned offset = (lane_id >> 2);

	// load b
	const T* const b_ptr = (lane_id & 0x1) ? db : b;

	frag_b.x[ 0] = detail::common::cast<half>(b_ptr[offset + 0]);
	frag_b.x[ 4] = detail::common::cast<half>(b_ptr[offset + 8]);
	frag_b.x[ 8] = frag_b.x[ 0];
	frag_b.x[12] = frag_b.x[ 4];
	if (CORRECTION_TERMS == 3 || (lane_id & 0x1) == 0) {
		frag_b.x[ 0 + 1] = frag_b.x[ 0];
		frag_b.x[ 4 + 1] = frag_b.x[ 4];
		frag_b.x[ 8 + 1] = frag_b.x[ 0];
		frag_b.x[12 + 1] = frag_b.x[ 4];
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

	if (lane_id & 0x2) return;

	const unsigned offset = (lane_id >> 2);

	frag_a.x[ 0] = detail::common::cast<half>(a_ptr[offset + 0]);
	frag_a.x[ 2] = detail::common::cast<half>(a_ptr[offset + 8]);
	frag_a.x[ 8] = frag_a.x[ 0];
	frag_a.x[10] = frag_a.x[ 2];
	if (CORRECTION_TERMS == 3 || (lane_id & 0x1) == 0) {
		{
			const auto a_fp32 = a_ptr[offset + 0];
			frag_a.x[ 0 + 1] = detail::common::cast<half>(a_fp32 - detail::common::cast<float>(detail::common::cast<half>(a_fp32)));
		}
		{
			const auto a_fp32 = a_ptr[offset + 8];
			frag_a.x[ 2 + 1] = detail::common::cast<half>(a_fp32 - detail::common::cast<float>(detail::common::cast<half>(a_fp32)));
		}
		frag_a.x[ 8 + 1] = frag_a.x[ 0 + 1];
		frag_a.x[10 + 1] = frag_a.x[ 2 + 1];
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

	if (lane_id & 0x2) return;

	// calculate offset
	const unsigned offset = (lane_id >> 2);

	const bool is_residual = (lane_id & 0x1);

	{
		const auto b_fp32 = b_ptr[offset + 0];
		frag_b.x[ 0] = detail::common::cast<half>(b_fp32);
		if (is_residual) {
			frag_b.x[ 0] = detail::common::cast<half>(b_fp32 - detail::common::cast<float>(frag_b.x[ 0]));
		}
	}
	{
		const auto b_fp32 = b_ptr[offset + 8];
		frag_b.x[ 4] = detail::common::cast<half>(b_fp32);
		if (is_residual) {
			frag_b.x[ 4] = detail::common::cast<half>(b_fp32 - detail::common::cast<float>(frag_b.x[ 4]));
		}
	}
	frag_b.x[ 8] = frag_b.x[ 0];
	frag_b.x[12] = frag_b.x[ 4];
	if (CORRECTION_TERMS == 3 || (lane_id & 0x1) == 0) {
		frag_b.x[ 0 + 1] = frag_b.x[ 0];
		frag_b.x[ 4 + 1] = frag_b.x[ 4];
		frag_b.x[ 8 + 1] = frag_b.x[ 0];
		frag_b.x[12 + 1] = frag_b.x[ 4];
	}
}
} // namespace sm_75
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

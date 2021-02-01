#ifndef __WMMAE_DETAIL_75_HPP__
#define __WMMAE_DETAIL_75_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_75 {
template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id >> 2;
	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = common::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 2] = common::cast<half>(ptr[index_offset + 8]);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id * 2;

	const bool load_flag = lane_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = common::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 1] = common::cast<half>(ptr[index_offset + 1]);
			frag.x[i * 8 + 4] = common::cast<half>(ptr[index_offset + 8]);
			frag.x[i * 8 + 5] = common::cast<half>(ptr[index_offset + 9]);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id * 2;

	const bool load_flag = lane_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = common::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 1] = common::cast<half>(ptr[index_offset + 1]);
			frag.x[i * 8 + 2] = common::cast<half>(ptr[index_offset + 8]);
			frag.x[i * 8 + 3] = common::cast<half>(ptr[index_offset + 9]);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id >> 2;

	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = common::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 4] = common::cast<half>(ptr[index_offset + 8]);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id >> 2;
	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = common::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 2] = common::cast<half>(ptr[index_offset + 8] * mul);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id * 2;

	const bool load_flag = lane_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = common::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 1] = common::cast<half>(ptr[index_offset + 1] * mul);
			frag.x[i * 8 + 4] = common::cast<half>(ptr[index_offset + 8] * mul);
			frag.x[i * 8 + 5] = common::cast<half>(ptr[index_offset + 9] * mul);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id * 2;

	const bool load_flag = lane_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = common::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 1] = common::cast<half>(ptr[index_offset + 1] * mul);
			frag.x[i * 8 + 2] = common::cast<half>(ptr[index_offset + 8] * mul);
			frag.x[i * 8 + 3] = common::cast<half>(ptr[index_offset + 9] * mul);
		}
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id >> 2;

	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = common::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 4] = common::cast<half>(ptr[index_offset + 8] * mul);
		}
	}
}

template <class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if ((tid & 0x3) == 0) {
			const auto mem_index = tid >> 2;
			ptr[mem_index + 0] = frag.x[0];
			ptr[mem_index + 8] = frag.x[2];
		}
	} else {
		if (!(tid & 0b11100)) {
			const auto mem_index = tid << 1;
			ptr[mem_index + 0] = frag.x[0];
			ptr[mem_index + 1] = frag.x[1];
			ptr[mem_index + 8] = frag.x[4];
			ptr[mem_index + 9] = frag.x[5];
		}
	}
}

template <class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if ((tid & 0x3) == 0) {
			const auto mem_index = tid >> 2;
			ptr[mem_index + 0] = frag.x[0] * mul;
			ptr[mem_index + 8] = frag.x[2] * mul;
		}
	} else {
		if (!(tid & 0b11100)) {
			const auto mem_index = tid << 1;
			ptr[mem_index + 0] = frag.x[0] * mul;
			ptr[mem_index + 1] = frag.x[1] * mul;
			ptr[mem_index + 8] = frag.x[4] * mul;
			ptr[mem_index + 9] = frag.x[5] * mul;
		}
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id >> 2) + ((lane_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id >> 2) << 4) + ((lane_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id >> 2) << 4) + ((lane_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id >> 2) + ((lane_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach_v(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned long index_offset = lane_id >> 2;
	const bool load_flag = (lane_id & 0x3) == 0;
	if(load_flag) {
		{
			const unsigned frag_index_list[2] = {0, 8};
			func(index_offset, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {2, 10};
			func(index_offset + 8, frag_index_list, 2);
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
			func(index_offset, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {1, 9};
			func(index_offset + 1, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {4, 12};
			func(index_offset + 8, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {5, 13};
			func(index_offset + 9, frag_index_list, 2);
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
			func(index_offset, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {1, 9};
			func(index_offset + 1, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {2, 10};
			func(index_offset + 8, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {3, 11};
			func(index_offset + 9, frag_index_list, 2);
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
			func(index_offset, frag_index_list, 2);
		}
		{
			const unsigned frag_index_list[2] = {4, 12};
			func(index_offset + 8, frag_index_list, 2);
		}
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id >> 2) + ((lane_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id >> 2) << 4) + ((lane_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id >> 2) << 4) + ((lane_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id >> 2) + ((lane_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
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

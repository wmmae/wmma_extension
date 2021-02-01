#ifndef __WMMAE_DETAIL_80_HPP__
#define __WMMAE_DETAIL_80_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_80 {
template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset]);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset]);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[8 ] = frag.x[0 ] = common::cast<half>(ptr[index_offset + 0 ]);
		frag.x[9 ] = frag.x[1 ] = common::cast<half>(ptr[index_offset + 1 ]);
		frag.x[12] = frag.x[4 ] = common::cast<half>(ptr[index_offset + 8 ]);
		frag.x[13] = frag.x[5 ] = common::cast<half>(ptr[index_offset + 9 ]);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset]);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[1  + index_offset]);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset]);
		frag.x[3  + 8] = frag.x[3 ] = common::cast<half>(ptr[9  + index_offset]);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	bool load_flag = (lane_id & 0x3) == 0;
	unsigned long index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset]);
		frag.x[4  + 8] = frag.x[4 ] = common::cast<half>(ptr[8  + index_offset]);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset] * mul);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset] * mul);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[8 ] = frag.x[0 ] = common::cast<half>(ptr[index_offset + 0 ] * mul);
		frag.x[9 ] = frag.x[1 ] = common::cast<half>(ptr[index_offset + 1 ] * mul);
		frag.x[12] = frag.x[4 ] = common::cast<half>(ptr[index_offset + 8 ] * mul);
		frag.x[13] = frag.x[5 ] = common::cast<half>(ptr[index_offset + 9 ] * mul);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset] * mul);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[1  + index_offset] * mul);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset] * mul);
		frag.x[3  + 8] = frag.x[3 ] = common::cast<half>(ptr[9  + index_offset] * mul);
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	bool load_flag = (lane_id & 0x3) == 0;
	unsigned long index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset] * mul);
		frag.x[4  + 8] = frag.x[4 ] = common::cast<half>(ptr[8  + index_offset] * mul);
	}
}

template <class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const bool load_flag = (lane_id & 0x3) == 0;
		const unsigned index_offset = lane_id >> 2;
		if (load_flag) {
			ptr[index_offset + 0 ] = common::cast<T>(frag.x[0 ]);
			ptr[index_offset + 8 ] = common::cast<T>(frag.x[2 ]);
		}
	} else {
		bool load_flag = (lane_id & 0b11100) == 0;
		const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
		if (load_flag) {
			ptr[index_offset + 0 ] = common::cast<T>(frag.x[0 ]);
			ptr[index_offset + 1 ] = common::cast<T>(frag.x[1 ]);
			ptr[index_offset + 8 ] = common::cast<T>(frag.x[4 ]);
			ptr[index_offset + 9 ] = common::cast<T>(frag.x[5 ]);
		}
	}
}

template <class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const bool load_flag = (lane_id & 0x3) == 0;
		const unsigned index_offset = lane_id >> 2;
		if (load_flag) {
			ptr[index_offset + 0 ] = common::cast<T>(frag.x[0 ] * mul);
			ptr[index_offset + 8 ] = common::cast<T>(frag.x[2 ] * mul);
		}
	} else {
		bool load_flag = (lane_id & 0b11100) == 0;
		const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
		if (load_flag) {
			ptr[index_offset + 0 ] = common::cast<T>(frag.x[0 ] * mul);
			ptr[index_offset + 1 ] = common::cast<T>(frag.x[1 ] * mul);
			ptr[index_offset + 8 ] = common::cast<T>(frag.x[4 ] * mul);
			ptr[index_offset + 9 ] = common::cast<T>(frag.x[5 ] * mul);
		}
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = ((tx & 0x1) << 4) + ((x & 0x2) << 2) + ((x & 0x4) << 5);
		func(x, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = (tx & 0x1) + ((tx & 0x2) << 6) + ((tx & 0x4) << 1);
		func(x, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = (tx & 0x1) + ((tx & 0x2) << 2) + ((tx & 0x4) << 5);
		func(x, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = ((tx & 0x1) << 4) + ((tx & 0x2) << 6) + ((tx & 0x4) << 1);
		func(x, start_index + offset);
	}
}

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

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = i & 0x7;
		const unsigned offset = ((tx & 0x1) << 4) + ((tx & 0x2) << 2) + ((tx & 0x4) << 5);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = i & 0x7;
		const unsigned offset = (tx & 0x1) + ((tx & 0x2) << 6) + ((tx & 0x4) << 1);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = i & 0x7;
		const unsigned offset = (tx & 0x1) + ((tx & 0x2) << 2) + ((tx & 0x4) << 5);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = i & 0x7;
		const unsigned offset = ((tx & 0x1) << 4) + ((tx & 0x2) << 6) + ((tx & 0x4) << 1);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
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

template <class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
	mtk::wmma::fill_zero(frag);
	add_eye(frag, mtk::wmma::detail::common::cast<T>(1.0f));
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

#ifndef __WMMAE_DETAIL_80_TF32_HPP__
#define __WMMAE_DETAIL_80_TF32_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_80 {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = ptr[0  + index_offset];
		frag.x[1 ] = ptr[8  + index_offset];
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = ptr[index_offset + 0 ];
		frag.x[2 ] = ptr[index_offset + 4 ];
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = ptr[index_offset + 0 ];
		frag.x[1 ] = ptr[index_offset + 4 ];
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = ptr[0  + index_offset];
		frag.x[2 ] = ptr[8  + index_offset];
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = ptr[0  + index_offset] * mul;
		frag.x[1 ] = ptr[8  + index_offset] * mul;
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = ptr[index_offset + 0 ] * mul;
		frag.x[2 ] = ptr[index_offset + 4 ] * mul;
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = ptr[index_offset + 0 ] * mul;
		frag.x[1 ] = ptr[index_offset + 4 ] * mul;
	}
}

template <class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = ptr[0  + index_offset] * mul;
		frag.x[2 ] = ptr[8  + index_offset] * mul;
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[0  + index_offset]);
		frag.x[1 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[8  + index_offset]);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 0 ]);
		frag.x[2 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 4 ]);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 0 ]);
		frag.x[1 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 4 ]);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[0  + index_offset]);
		frag.x[2 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[8  + index_offset]);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[0  + index_offset] * mul);
		frag.x[1 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[8  + index_offset] * mul);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 0 ] * mul);
		frag.x[2 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 4 ] * mul);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0b11100) == 0;
	const unsigned index_offset = (lane_id & 0x3) + ((lane_id & 0x4) << 1);
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 0 ] * mul);
		frag.x[1 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[index_offset + 4 ] * mul);
	}
}

template <class T>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[0  + index_offset] * mul);
		frag.x[2 ] = mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(ptr[8  + index_offset] * mul);
	}
}


template <class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, T>& frag, const nvcuda::wmma::layout_t layout) {
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
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
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
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 4) + (lane_id >> 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x2) << 5) + ((x & 0x1) << 3);
		func(x, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id & 0x3) + ((lane_id & 0x1c) << 1);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 6) + ((x & 0x2) << 1);
		func(x, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id & 0x3) + ((lane_id & 0x1c) << 1);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 2) + ((x & 0x2) << 5);
		func(x, start_index + offset);
	}
}

template <class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 4) + (lane_id >> 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 6) + ((x & 0x2) << 2);
		func(x, start_index + offset);
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
			func(index_offset, frag_index_list, 1);
		}
		{
			const unsigned frag_index_list[1] = {1};
			func(index_offset + 8, frag_index_list, 1);
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
			func(index_offset, frag_index_list, 1);
		}
		{
			const unsigned frag_index_list[1] = {2};
			func(index_offset + 4, frag_index_list, 1);
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
			func(index_offset, frag_index_list, 1);
		}
		{
			const unsigned frag_index_list[1] = {1};
			func(index_offset + 4, frag_index_list, 1);
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
			func(index_offset, frag_index_list, 1);
		}
		{
			const unsigned frag_index_list[1] = {2};
			func(index_offset + 8, frag_index_list, 1);
		}
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 4) + (lane_id >> 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x2) << 5) + ((x & 0x1) << 3);
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id & 0x3) + ((lane_id & 0x1c) << 1);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 6) + ((x & 0x2) << 1);
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = (lane_id & 0x3) + ((lane_id & 0x1c) << 1);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 2) + ((x & 0x2) << 5);
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 4) + (lane_id >> 2);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x & 0x1) << 6) + ((x & 0x2) << 2);
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
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

template <class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, T>& frag) {
	mtk::wmma::fill_zero(frag);
	add_eye(frag, 1.0f);
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

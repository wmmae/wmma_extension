#ifndef __WMMAE_DETAIL_80_HPP__
#define __WMMAE_DETAIL_80_HPP__
#include <mma.h>
#include "common.hpp"
namespace mtk {
namespace wmma {
namespace detail {
namespace sm_80 {
template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset]);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[16 + index_offset]);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset]);
		frag.x[3  + 8] = frag.x[3 ] = common::cast<half>(ptr[24 + index_offset]);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x8) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[index_offset + 8 ] = frag.x[index_offset + 0 ] = common::cast<half>(ptr[0 ]);
		frag.x[index_offset + 9 ] = frag.x[index_offset + 1 ] = common::cast<half>(ptr[1 ]);
		frag.x[index_offset + 12] = frag.x[index_offset + 4 ] = common::cast<half>(ptr[8 ]);
		frag.x[index_offset + 13] = frag.x[index_offset + 5 ] = common::cast<half>(ptr[9 ]);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x8) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset]);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[1  + index_offset]);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset]);
		frag.x[3  + 8] = frag.x[3 ] = common::cast<half>(ptr[9  + index_offset]);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	bool load_flag = (lane_id & 0x3) == 0;
	unsigned long index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset]);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[16 + index_offset]);
		frag.x[4  + 8] = frag.x[4 ] = common::cast<half>(ptr[8  + index_offset]);
		frag.x[5  + 8] = frag.x[5 ] = common::cast<half>(ptr[24 + index_offset]);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset] * mul);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[16 + index_offset] * mul);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset] * mul);
		frag.x[3  + 8] = frag.x[3 ] = common::cast<half>(ptr[24 + index_offset] * mul);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x8) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[index_offset + 8 ] = frag.x[index_offset + 0 ] = common::cast<half>(ptr[0 ] * mul);
		frag.x[index_offset + 9 ] = frag.x[index_offset + 1 ] = common::cast<half>(ptr[1 ] * mul);
		frag.x[index_offset + 12] = frag.x[index_offset + 4 ] = common::cast<half>(ptr[8 ] * mul);
		frag.x[index_offset + 13] = frag.x[index_offset + 5 ] = common::cast<half>(ptr[9 ] * mul);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x8) == 0;
	const unsigned index_offset = ((lane_id & 0x3) << 1) + ((lane_id & 0x4) << 2);
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset] * mul);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[1  + index_offset] * mul);
		frag.x[2  + 8] = frag.x[2 ] = common::cast<half>(ptr[8  + index_offset] * mul);
		frag.x[3  + 8] = frag.x[3 ] = common::cast<half>(ptr[9  + index_offset] * mul);
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	const bool load_flag = (lane_id & 0x3) == 0;
	const unsigned index_offset = lane_id >> 2;
	if(load_flag) {
		frag.x[0  + 8] = frag.x[0 ] = common::cast<half>(ptr[0  + index_offset] * mul);
		frag.x[1  + 8] = frag.x[1 ] = common::cast<half>(ptr[16 + index_offset] * mul);
		frag.x[4  + 8] = frag.x[4 ] = common::cast<half>(ptr[8  + index_offset] * mul);
		frag.x[5  + 8] = frag.x[5 ] = common::cast<half>(ptr[24 + index_offset] * mul);
	}
	__syncthreads();
}

template <class T>
__device__ inline void store_vector_sync(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const bool load_flag = (lane_id & 0x8) == 0;
		const unsigned index_offset = lane_id >> 2;
		if (load_flag) {
			ptr[index_offset + 0 ] = common::cast<T>(frag.x[0 ]);
			ptr[index_offset + 16] = common::cast<T>(frag.x[1 ]);
			ptr[index_offset + 8 ] = common::cast<T>(frag.x[2 ]);
			ptr[index_offset + 24] = common::cast<T>(frag.x[3 ]);
		}
	} else {
		bool load_flag = (lane_id & 0x3) == 0;
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
__device__ inline void store_vector_sync(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (layout == nvcuda::wmma::mem_col_major) {
		const bool load_flag = (lane_id & 0x8) == 0;
		const unsigned index_offset = lane_id >> 2;
		if (load_flag) {
			ptr[index_offset + 0 ] = common::cast<T>(frag.x[0 ] * mul);
			ptr[index_offset + 16] = common::cast<T>(frag.x[1 ] * mul);
			ptr[index_offset + 8 ] = common::cast<T>(frag.x[2 ] * mul);
			ptr[index_offset + 24] = common::cast<T>(frag.x[3 ] * mul);
		}
	} else {
		bool load_flag = (lane_id & 0x3) == 0;
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

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = ((tx & 0x1) << 4) + ((x & 0x2) << 2) + ((x & 0x4) << 5);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = (tx & 0x1) + ((tx & 0x2) << 6) + ((tx & 0x4) << 1);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 1) + ((lane_id & 0x1c) << 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = (tx & 0x1) + ((tx & 0x2) << 2) + ((tx & 0x4) << 5);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned start_index = ((lane_id & 0x3) << 5) + (lane_id >> 2);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned tx = x & 0x7;
		const unsigned offset = ((tx & 0x1) << 4) + ((tx & 0x2) << 6) + ((tx & 0x4) << 1);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}
} // namespace sm_80
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

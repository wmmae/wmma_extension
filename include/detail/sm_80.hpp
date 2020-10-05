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
} // namespace sm_80
} // namespace detail
} // namespace wmma
} // namespace mtk
#endif

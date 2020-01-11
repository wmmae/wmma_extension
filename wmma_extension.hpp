#ifndef __WMMA_EXTENSION_HPP__
#define __WMMA_EXTENSION_HPP__
#include <mma.h>

namespace mtk {
namespace wmma {
namespace utils {
template <class T> inline __device__ T cast(const float v);
template <class T> inline __device__ T cast(const half v);
template <> inline __device__ float cast(const float v){return v;}
template <> inline __device__ float cast(const half v){return __half2float(v);}
template <> inline __device__ half cast(const float v){return __float2half(v);}
template <> inline __device__ half cast(const half v){return v;}
} // namespace utils

// Common for sm_70 and sm_75
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, float, Layout>& frag) {
	int4* const i4 = reinterpret_cast<int4*>(frag.x);
	const unsigned size = sizeof(float) * nvcuda::wmma::fragment<Use, M, N, K, float, Layout>::num_elements;
	for (unsigned i = 0; i < size / sizeof(int4); i++) {
		i4[i] = make_int4(0, 0, 0, 0);
	}
}
template <class Use, int M, int N, int K, class Layout>
__device__ inline void fill_zero(nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& frag) {
	int4* const i4 = reinterpret_cast<int4*>(frag.x);
	const unsigned size = sizeof(half) * nvcuda::wmma::fragment<Use, M, N, K, half, Layout>::num_elements;
	for (unsigned i = 0; i < size / sizeof(int4); i++) {
		i4[i] = make_int4(0, 0, 0, 0);
	}
}


// For sm75
template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id >> 2;
	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = utils::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 2] = utils::cast<half>(ptr[index_offset + 8]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id * 2;

	bool load_flag = warp_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = utils::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 1] = utils::cast<half>(ptr[index_offset + 1]);
			frag.x[i * 8 + 4] = utils::cast<half>(ptr[index_offset + 8]);
			frag.x[i * 8 + 5] = utils::cast<half>(ptr[index_offset + 9]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id * 2;

	bool load_flag = warp_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = utils::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 1] = utils::cast<half>(ptr[index_offset + 1]);
			frag.x[i * 8 + 2] = utils::cast<half>(ptr[index_offset + 8]);
			frag.x[i * 8 + 3] = utils::cast<half>(ptr[index_offset + 9]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id >> 2;

	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = utils::cast<half>(ptr[index_offset]);
			frag.x[i * 8 + 4] = utils::cast<half>(ptr[index_offset + 8]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id >> 2;
	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = utils::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 2] = utils::cast<half>(ptr[index_offset + 8] * mul);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id * 2;

	bool load_flag = warp_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = utils::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 1] = utils::cast<half>(ptr[index_offset + 1] * mul);
			frag.x[i * 8 + 4] = utils::cast<half>(ptr[index_offset + 8] * mul);
			frag.x[i * 8 + 5] = utils::cast<half>(ptr[index_offset + 9] * mul);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id * 2;

	bool load_flag = warp_id < 4;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8    ] = utils::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 1] = utils::cast<half>(ptr[index_offset + 1] * mul);
			frag.x[i * 8 + 2] = utils::cast<half>(ptr[index_offset + 8] * mul);
			frag.x[i * 8 + 3] = utils::cast<half>(ptr[index_offset + 9] * mul);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = warp_id >> 2;

	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 2; i++) {
			frag.x[i * 8] = utils::cast<half>(ptr[index_offset] * mul);
			frag.x[i * 8 + 4] = utils::cast<half>(ptr[index_offset + 8] * mul);
		}
	}
	__syncthreads();
}

template <class Func>
__device__ inline void foreach_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (warp_id >> 2) + ((warp_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = ((warp_id >> 2) << 4) + ((warp_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = ((warp_id >> 2) << 4) + ((warp_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
}

template <class Func>
__device__ inline void foreach_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (warp_id >> 2) + ((warp_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		func(i, index);
		func(j, index);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = warp_id & 0xf;
	const unsigned start_index = (warp_id >> 2) + ((warp_id & 0b11) << 5);
	for (std::size_t x = 0; x < (frag.num_elements >> 1); x++) {
		const unsigned i = (x + skew) & 0x7;
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0x3) + ((warp_id & 0xf) >> 3) + (warp_id >> 4) << 2;
	const unsigned start_index = ((warp_id >> 2) << 4) + ((warp_id & 0b11) << 1);
	for (std::size_t x = 0; x < (frag.num_elements >> 1); x++) {
		const unsigned i = (x + skew) & 0xf;
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0x3) + ((warp_id & 0xf) >> 3) + (warp_id >> 4) << 2;
	const unsigned start_index = ((warp_id >> 2) << 4) + ((warp_id & 0b11) << 1);
	for (std::size_t x = 0; x < (frag.num_elements >> 1); x++) {
		const unsigned i = (x + skew) & 0xf;
		const unsigned offset = (i & 0b1) + ((i & 0b10) << 2) + ((i & 0b100) << 5);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm75(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = warp_id & 0xf;
	const unsigned start_index = (warp_id >> 2) + ((warp_id & 0b11) << 5);
	for (std::size_t x = 0; x < (frag.num_elements >> 1); x++) {
		const unsigned i = (x + skew) & 0x7;
		const unsigned offset = ((i & 0b1) << 4) + ((i & 0b10) << 6) + ((i & 0b100) << 1);
		const unsigned index = start_index + offset;
		const unsigned j = i + (frag.num_elements >> 1);
		const auto v = ptr[(index >> 4) * ldm + (index & 0xf)];
		frag.x[i] = func(i, v);
		frag.x[j] = func(j, v);
	}
	__syncthreads();
}

template <class T>
__device__ inline void make_identity_matrix_sm75(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	if (std::is_same<T, float>::value) {
		int4* const i4 = reinterpret_cast<int4*>(frag.x);
		i4[0] = make_int4(0, 0, 0, 0);
		i4[1] = make_int4(0, 0, 0, 0);
	} else {
		int4* const i4 = reinterpret_cast<int4*>(frag.x);
		i4[0] = make_int4(0, 0, 0, 0);
	}

	const unsigned mod9 = warp_id % 9;

	unsigned index_offset = mod9 >> 2;
	bool set_flag = mod9 == 0 || mod9 == 4;

	if(set_flag) {
		frag.x[index_offset] = frag.x[index_offset + 6] = utils::cast<T>(1.0f);
	}
}

// For sm70
template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = ((warp_id >> 4) << 2) + (((warp_id >> 2) & 0x1) << 3);
	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = (warp_id & 0x3) << 4;

	bool load_flag = (warp_id == 0) || (warp_id == 4);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = (warp_id & 0x3) << 4;

	bool load_flag = (warp_id == 0) || (warp_id == 4);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = ((warp_id >> 4) << 2) + (((warp_id >> 3) & 0x1) << 3);
	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = ((warp_id >> 4) << 2) + (((warp_id >> 2) & 0x1) << 3);
	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset] * mul);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = (warp_id & 0x3) << 4;

	bool load_flag = (warp_id == 0) || (warp_id == 4);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset] * mul);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = (warp_id & 0x3) << 4;

	bool load_flag = (warp_id == 0) || (warp_id == 4);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset] * mul);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const T mul, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = ((warp_id >> 4) << 2) + (((warp_id >> 3) & 0x1) << 3);
	bool load_flag = (warp_id & 0x3) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			frag.x[i] = utils::cast<half>(ptr[i + index_offset] * mul);
		}
	}
	__syncthreads();
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0x7) + ((warp_id & 0xf) >> 3);
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0xf) + (warp_id >> 4);
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = x;
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0xf) + (warp_id >> 4);
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = x;
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0x7) + ((warp_id & 0xf) >> 3);
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0x7) + ((warp_id & 0xf) >> 3);
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0xf) + (warp_id >> 4);
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = x;
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0xf) + (warp_id >> 4);
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = x;
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned skew = (warp_id & 0x7) + ((warp_id & 0xf) >> 3);
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned x = (i + skew) & 0xf;
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;
		frag.x[x] = func(x, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T>
__device__ inline void make_identity_matrix_sm70(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	if (std::is_same<T, float>::value) {
		int4* const i4 = reinterpret_cast<int4*>(frag.x);
		i4[0] = make_int4(0, 0, 0, 0);
		i4[1] = make_int4(0, 0, 0, 0);
	} else {
		int4* const i4 = reinterpret_cast<int4*>(frag.x);
		i4[0] = make_int4(0, 0, 0, 0);
	}

	unsigned index_offset = 0;
	if(warp_id >> 4) {
		index_offset = 4;
	}

	const unsigned p0 = (warp_id >> 2) & 0x3;
	if(p0 == 0 || p0 == 3) {
		frag.x[(warp_id & 0x3) + index_offset] = utils::cast<T>(1.0f);
	}
	__syncthreads();
}

// arch switch
template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, const bool fill = true) {
#if __CUDA_ARCH__ < 710
	load_vector_sync_sm70(frag, ptr, fill);
#else
	load_vector_sync_sm75(frag, ptr, fill);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, const T mul, const bool fill = true) {
#if __CUDA_ARCH__ < 710
	load_vector_sync_sm70(frag, ptr, mul, fill);
#else
	load_vector_sync_sm75(frag, ptr, mul, fill);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, Func func) {
#if __CUDA_ARCH__ < 710
	foreach_sm70(frag, func);
#else
	foreach_sm75(frag, func);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, unsigned ldm, Func func) {
#if __CUDA_ARCH__ < 710
	load_matrix_with_operation_sync_sm70(frag, ptr, ldm, func);
#else
	load_matrix_with_operation_sync_sm75(frag, ptr, ldm, func);
#endif
}

template <class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
#if __CUDA_ARCH__ < 710
	make_identity_matrix_sm70(frag);
#else
	make_identity_matrix_sm75(frag);
#endif
}
} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

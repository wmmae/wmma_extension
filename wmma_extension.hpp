#ifndef __WMMA_EXTENSION_HPP__
#define __WMMA_EXTENSION_HPP__
#include <mma.h>
#include <stdio.h>

namespace mtk {
namespace wmma {
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
} // namespace wmma

namespace detail {
namespace utils {
template <class T> inline __device__ T cast(const float v);
template <class T> inline __device__ T cast(const half v);
template <> inline __device__ float cast(const float v){return v;}
template <> inline __device__ float cast(const half v){return __half2float(v);}
template <> inline __device__ half cast(const float v){return __float2half(v);}
template <> inline __device__ half cast(const half v){return v;}
} // namespace utils

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

template <class T>
__device__ inline void store_vector_sync_sm75(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout) {
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
__device__ inline void store_vector_sync_sm75(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
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
	const unsigned start_index = (warp_id >> 2) + ((warp_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
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
	const unsigned start_index = ((warp_id >> 2) << 4) + ((warp_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
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
	const unsigned start_index = ((warp_id >> 2) << 4) + ((warp_id & 0b11) << 1);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
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
	const unsigned start_index = (warp_id >> 2) + ((warp_id & 0b11) << 5);
	for (std::size_t i = 0; i < (frag.num_elements >> 1); i++) {
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

template <class T>
__device__ inline void add_eye_sm75(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned mod9 = warp_id % 9;

	unsigned index_offset = mod9 >> 2;
	bool set_flag = mod9 == 0 || mod9 == 4;

	if(set_flag) {
		frag.x[index_offset] += alpha;
		frag.x[index_offset + 6] += alpha;
	}
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment_sm75(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag_a,
		const T* const a, const S* const da,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_a);
	}
	const unsigned warp_id = threadIdx.x & 0x1f;

	if (warp_id & 0x2) return;

	// load a
	const unsigned offset = (warp_id >> 2);

	frag_a.x[ 0] = detail::utils::cast<half>(a[offset + 0]);
	frag_a.x[ 2] = detail::utils::cast<half>(a[offset + 8]);
	frag_a.x[ 8] = frag_a.x[ 0];
	frag_a.x[10] = frag_a.x[ 2];
	if (CORRECTION_TERMS == 3 || (warp_id & 0x1) == 0) {
		frag_a.x[ 0 + 1] = detail::utils::cast<half>(da[offset + 0]);
		frag_a.x[ 2 + 1] = detail::utils::cast<half>(da[offset + 8]);
		frag_a.x[ 8 + 1] = frag_a.x[ 0 + 1];
		frag_a.x[10 + 1] = frag_a.x[ 2 + 1];
	}
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment_sm75(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag_b,
		const T* const b, const S* const db,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_b);
	}
	const unsigned warp_id = threadIdx.x & 0x1f;

	if (warp_id & 0x2) return;

	// load a
	const unsigned offset = (warp_id >> 2);

	// load b
	const T* const b_ptr = (warp_id & 0x1) ? db : b;

	frag_b.x[ 0] = detail::utils::cast<half>(b_ptr[offset + 0]);
	frag_b.x[ 4] = detail::utils::cast<half>(b_ptr[offset + 8]);
	frag_b.x[ 8] = frag_b.x[ 0];
	frag_b.x[12] = frag_b.x[ 4];
	if (CORRECTION_TERMS == 3 || (warp_id & 0x1) == 0) {
		frag_b.x[ 0 + 1] = frag_b.x[ 0];
		frag_b.x[ 4 + 1] = frag_b.x[ 4];
		frag_b.x[ 8 + 1] = frag_b.x[ 0];
		frag_b.x[12 + 1] = frag_b.x[ 4];
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

	bool load_flag = (warp_id == 0) || (warp_id == 8);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<half>(ptr[i]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ inline void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const bool fill) {
	if (fill)
		nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const unsigned warp_id = threadIdx.x & 0x1f;

	bool load_flag = (warp_id == 0) || (warp_id == 4);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<half>(ptr[i]);
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

template <class T>
__device__ inline void store_vector_sync_sm70(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const nvcuda::wmma::layout_t layout) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if (!(tid & 0b01000)) {
			const auto mem_index = ((tid & 0b10000) >> 2) + (tid & 0x3) + ((tid & 0x4) << 1);
			ptr[mem_index + 0] = frag.x[0];
		}
	} else {
		if (tid == 0 || tid == 8) {
			const auto mem_index = tid;
			ptr[mem_index + 0] = frag.x[0];
			ptr[mem_index + 1] = frag.x[1];
			ptr[mem_index + 2] = frag.x[2];
			ptr[mem_index + 3] = frag.x[3];
			ptr[mem_index + 4] = frag.x[4];
			ptr[mem_index + 5] = frag.x[5];
			ptr[mem_index + 6] = frag.x[6];
			ptr[mem_index + 7] = frag.x[7];
		}
	}
}

// partial specialization
template <>
__device__ inline void store_vector_sync_sm70<float>(float* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& frag, const nvcuda::wmma::layout_t layout) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if (!(tid & 0b10) && !(tid & 0b1000)) {
			const auto mem_index = ((tid & 0b10000) >> 2) + (tid & 0b1) + ((tid & 0b100) << 1);
			ptr[mem_index + 0] = frag.x[0];
			ptr[mem_index + 2] = frag.x[2];
		}
	} else {
		if (!(tid & 0b1) && !(tid & 0b10000) && !(tid & 0b100)) {
			const auto mem_index = tid;
			ptr[mem_index + 0] = frag.x[0];
			ptr[mem_index + 1] = frag.x[1];
			ptr[mem_index + 4] = frag.x[4];
			ptr[mem_index + 5] = frag.x[5];
		}
	}
}

template <class T>
__device__ inline void store_vector_sync_sm70(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if (!(tid & 0b01000)) {
			const auto mem_index = ((tid & 0b10000) >> 2) + (tid & 0x3) + ((tid & 0x4) << 1);
			ptr[mem_index + 0] = frag.x[0] * mul;
		}
	} else {
		if (tid == 0 || tid == 8) {
			const auto mem_index = tid;
			ptr[mem_index + 0] = frag.x[0] * mul;
			ptr[mem_index + 1] = frag.x[1] * mul;
			ptr[mem_index + 2] = frag.x[2] * mul;
			ptr[mem_index + 3] = frag.x[3] * mul;
			ptr[mem_index + 4] = frag.x[4] * mul;
			ptr[mem_index + 5] = frag.x[5] * mul;
			ptr[mem_index + 6] = frag.x[6] * mul;
			ptr[mem_index + 7] = frag.x[7] * mul;
		}
	}
}

// partial specialization
template <>
__device__ inline void store_vector_sync_sm70<float>(float* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& frag, const float mul, const nvcuda::wmma::layout_t layout) {
	const auto tid = threadIdx.x & 0x1f;
	if (layout == nvcuda::wmma::mem_col_major) {
		if (!(tid & 0b10) && !(tid & 0b1000)) {
			const auto mem_index = ((tid & 0b10000) >> 2) + (tid & 0b1) + ((tid & 0b100) << 1);
			ptr[mem_index + 0] = frag.x[0] * mul;
			ptr[mem_index + 2] = frag.x[2] * mul;
		}
	} else {
		if (!(tid & 0b1) && !(tid & 0b10000) && !(tid & 0b100)) {
			const auto mem_index = tid;
			ptr[mem_index + 0] = frag.x[0] * mul;
			ptr[mem_index + 1] = frag.x[1] * mul;
			ptr[mem_index + 4] = frag.x[4] * mul;
			ptr[mem_index + 5] = frag.x[5] * mul;
		}
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = x;
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = x;
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class Func>
__device__ inline void foreach_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t x = 0; x < frag.num_elements; x++) {
		const unsigned offset = ((x >> 2) << 6) + (x & 0b11);
		const unsigned index = start_index + offset;
		func(x, index);
	}
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned offset = ((i >> 2) << 6) + (i & 0b11);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 2) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned offset = i;
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 7) + ((warp_id >> 4) << 6) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned offset = i;
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
	}
	__syncthreads();
}

template <class T, class Func>
__device__ inline void load_matrix_with_operation_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr, const unsigned ldm, Func func) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	const unsigned start_index = (((warp_id >> 3) & 0b1) << 3) + ((warp_id >> 4) << 2) + ((warp_id & 0b11) << 4);
	for (std::size_t i = 0; i < frag.num_elements; i++) {
		const unsigned offset = ((i >> 2) << 6) + (i & 0b11);
		const unsigned index = start_index + offset;
		frag.x[i] = func(i, ptr[(index >> 4) * ldm + (index & 0xf)]);
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

template <class T>
__device__ inline void add_eye_sm70(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
	const unsigned warp_id = threadIdx.x & 0x1f;
	unsigned index_offset = 0;
	if(warp_id >> 4) {
		index_offset = 4;
	}

	const unsigned p0 = (warp_id >> 2) & 0x3;
	if(p0 == 0 || p0 == 3) {
		frag.x[(warp_id & 0x3) + index_offset] += alpha;
	}
	__syncthreads();
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment_sm70(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag_a,
		const T* const a, const S* const da,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_a);
	}
	const unsigned warp_id = threadIdx.x & 0x1f;

	if (CORRECTION_TERMS == 2 && (warp_id & 0x3) == 0x3) {
		return;
	}

	const T* const a_ptr = ((warp_id & 0x1) == 0) ? a : da;
	const unsigned a_offset = ((warp_id & 0x10) >> 2) + ((warp_id & 0x4) << 1);

	frag_a.x[0] = detail::utils::cast<half>(a_ptr[a_offset + 0]);
	frag_a.x[1] = detail::utils::cast<half>(a_ptr[a_offset + 1]);
	frag_a.x[2] = detail::utils::cast<half>(a_ptr[a_offset + 2]);
	frag_a.x[3] = detail::utils::cast<half>(a_ptr[a_offset + 3]);
}

template <class T, class S, unsigned CORRECTION_TERMS = 2>
__device__ inline void make_direct_product_fragment_sm70(
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag_b,
		const T* const b, const S* const db,
		const bool fill
		) {
	if (fill) {
		mtk::wmma::fill_zero(frag_b);
	}
	const unsigned warp_id = threadIdx.x & 0x1f;

	if (CORRECTION_TERMS == 2 && (warp_id & 0x3) == 0x3) {
		return;
	}

	const T* const b_ptr = ((warp_id & 0x2) != 0) ? b : db;
	const unsigned b_offset = ((warp_id & 0x10) >> 2) + (warp_id & 0x8);

	frag_b.x[0] = detail::utils::cast<half>(b_ptr[b_offset + 0]);
	frag_b.x[1] = detail::utils::cast<half>(b_ptr[b_offset + 1]);
	frag_b.x[2] = detail::utils::cast<half>(b_ptr[b_offset + 2]);
	frag_b.x[3] = detail::utils::cast<half>(b_ptr[b_offset + 3]);
}
} // namespace detail

namespace wmma {

// arch switch
template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, const bool fill = true) {
#if __CUDA_ARCH__ < 710
	detail::load_vector_sync_sm70(frag, ptr, fill);
#else
	detail::load_vector_sync_sm75(frag, ptr, fill);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, const T mul, const bool fill = true) {
#if __CUDA_ARCH__ < 710
	detail::load_vector_sync_sm70(frag, ptr, mul, fill);
#else
	detail::load_vector_sync_sm75(frag, ptr, mul, fill);
#endif
}

template <int M, int N, int K, class T>
__device__ inline void store_vector_sync(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const nvcuda::wmma::layout_t layout) {
#if __CUDA_ARCH__ < 710
	detail::store_vector_sync_sm70(ptr, frag, layout);
#else
	detail::store_vector_sync_sm75(ptr, frag, layout);
#endif
}

template <int M, int N, int K, class T>
__device__ inline void store_vector_sync(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
#if __CUDA_ARCH__ < 710
	detail::store_vector_sync_sm70(ptr, frag, mul, layout);
#else
	detail::store_vector_sync_sm75(ptr, frag, mul, layout);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, Func func) {
#if __CUDA_ARCH__ < 710
	detail::foreach_sm70(frag, func);
#else
	detail::foreach_sm75(frag, func);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, unsigned ldm, Func func) {
#if __CUDA_ARCH__ < 710
	detail::load_matrix_with_operation_sync_sm70(frag, ptr, ldm, func);
#else
	detail::load_matrix_with_operation_sync_sm75(frag, ptr, ldm, func);
#endif
}

template <class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
#if __CUDA_ARCH__ < 710
	detail::make_identity_matrix_sm70(frag);
#else
	detail::make_identity_matrix_sm75(frag);
#endif
}

template <class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
#if __CUDA_ARCH__ < 710
	detail::add_eye_sm70(frag, alpha);
#else
	detail::add_eye_sm75(frag, alpha);
#endif
}


template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
#if __CUDA_ARCH__ < 710
	detail::make_direct_product_fragment_sm70<T, S, 2>(frag_x, x, dx, fill);
#else
	detail::make_direct_product_fragment_sm75<T, S, 2>(frag_x, x, dx, fill);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
#if __CUDA_ARCH__ < 710
	detail::make_direct_product_fragment_sm70<T, S, 3>(frag_x, x, dx, fill);
#else
	detail::make_direct_product_fragment_sm75<T, S, 3>(frag_x, x, dx, fill);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const nvcuda::wmma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const char* name = "") {
	if ((threadIdx.x & 0x1f) == 0) {
		if (name[0] != '\0') {
			printf("%s = \n", name);
		}
	}
	for (unsigned i = 0; i < warpSize; i++) {
		if (i == (threadIdx.x & 0x1f)) {
			for (unsigned j = 0; j < frag.num_elements; j++) {
				const auto v = mtk::detail::utils::cast<float>(frag.x[j]);
				if (v == 0.0f) {
					printf(" %.3e ", 0.0f);
				} else if (v > 0) {
					printf(" %.3e ", v);
				} else {
					printf("%.3e ", v);
				}
			}
			printf("\n");
		}
		__syncthreads();
	}
}

} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

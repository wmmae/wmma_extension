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
template <class T>
__device__ void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr) {
	nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const auto warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = ((warp_id >> 4) << 2) + (((warp_id >> 2) & 0x1) << 3) + ((warp_id & 0x3) << 4);
	bool load_flag = (warp_id & 0x2) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			frag.x[i] = utils::cast<T>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr) {
	nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const auto warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = (warp_id & 0x3) << 4;

	bool load_flag = (warp_id == 0) || (warp_id == 1) || (warp_id == 4) || (warp_id == 5);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<T>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ void load_vector_sync_sm70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>& frag, const T* const ptr) {
	nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const auto warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = (warp_id & 0x3) << 4;

	bool load_flag = (warp_id == 0) || (warp_id == 1) || (warp_id == 4) || (warp_id == 5);
	if(load_flag) {
		for(unsigned i = 0; i < 16; i++) {
			frag.x[i] = utils::cast<T>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}

template <class T>
__device__ void load_vector_sync_sm_70(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& frag, const T* const ptr) {
	nvcuda::wmma::fill_fragment(frag, __float2half(0));
	const auto warp_id = threadIdx.x & 0x1f;
	unsigned long index_offset = ((warp_id >> 4) << 2) + (((warp_id >> 2) & 0x1) << 3) + ((warp_id & 0x3) << 4);
	bool load_flag = (warp_id & 0x2) == 0;
	if(load_flag) {
		for(unsigned i = 0; i < 4; i++) {
			frag.x[i] = utils::cast<T>(ptr[i + index_offset]);
		}
	}
	__syncthreads();
}
} // namespace wmma
} // namespace mtk

#ifndef __WMMA_EXTENSION_HPP__
#define __WMMA_EXTENSION_HPP__
#include <mma.h>
#include "detail/sm_70.hpp"
#include "detail/sm_75.hpp"
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

// For sm75

// For sm70
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

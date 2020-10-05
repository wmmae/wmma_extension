#ifndef __WMMA_EXTENSION_HPP__
#define __WMMA_EXTENSION_HPP__
#include <stdio.h>
#include <mma.h>
#include <stdio.h>
#if defined(__CUDA_ARCH__)
#include "detail/m8n8k4.hpp"
#include "detail/common.hpp"

#if __CUDA_ARCH__ >= 700
#include "detail/sm_70.hpp"
#endif
#if __CUDA_ARCH__ >= 710
#include "detail/sm_75.hpp"
#endif
#if __CUDA_ARCH__ >= 800
#include "detail/sm_80.hpp"
#include "detail/sm_80_tf32.hpp"
#endif

namespace mtk {
// arch switch
#if __CUDA_ARCH__ < 710
namespace detail_namespace = mtk::wmma::detail::sm_70;
#elif __CUDA_ARCH__ < 800
namespace detail_namespace = mtk::wmma::detail::sm_75;
#else
namespace detail_namespace = mtk::wmma::detail::sm_80;
#endif

namespace wmma {
template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, const bool fill = true) {
	detail_namespace::load_vector_sync(frag, ptr, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void load_vector_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, const T mul, const bool fill = true) {
	detail_namespace::load_vector_sync(frag, ptr, mul, fill);
}

template <int M, int N, int K, class T>
__device__ inline void store_vector_sync(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const nvcuda::wmma::layout_t layout) {
	detail_namespace::store_vector_sync(ptr, frag, layout);
}

template <int M, int N, int K, class T>
__device__ inline void store_vector_sync(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	detail_namespace::store_vector_sync(ptr, frag, mul, layout);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class Func>
__device__ inline void foreach(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, Func func) {
	detail_namespace::foreach(frag, func);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class Func>
__device__ inline void load_matrix_with_operation_sync(nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag, const T* const ptr, unsigned ldm, Func func) {
	detail_namespace::load_matrix_with_operation_sync(frag, ptr, ldm, func);
}

template <class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag) {
	detail_namespace::make_identity_matrix(frag);
}

template <class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T>& frag, const T alpha) {
	detail_namespace::add_eye(frag, alpha);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<T, S, 2>(frag_x, x, dx, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<T, S, 3>(frag_x, x, dx, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag_x,
		const float* const x,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<2>(frag_x, x, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, half, MemMajor>& frag_x,
		const float* const x,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<3>(frag_x, x, fill);
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
				const auto v = mtk::wmma::detail::common::cast<float>(frag.x[j]);
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
#endif /* defined(__CUDA_ARCH__)*/

#endif /* end of include guard */

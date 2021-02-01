#ifndef __WMMA_EXTENSION_HPP__
#define __WMMA_EXTENSION_HPP__
#include <stdio.h>
#include <mma.h>
#include <stdio.h>
#include "detail/m8n8k4.hpp"
#include "detail/common.hpp"

#include "detail/sm_70.hpp"
#include "detail/sm_75.hpp"
#include "detail/sm_80.hpp"
#include "detail/sm_80_tf32.hpp"

namespace mtk {
namespace wmma {
// arch switch
#if __CUDA_ARCH__ < 710
namespace detail_namespace = mtk::wmma::detail::sm_70;
#elif __CUDA_ARCH__ < 800
namespace detail_namespace = mtk::wmma::detail::sm_75;
#else
namespace detail_namespace = mtk::wmma::detail::sm_80;
#endif

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class FT>
__device__ inline void load_vector(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, const T* const ptr, const bool fill = true) {
	detail_namespace::load_vector(frag, ptr, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class FT>
__device__ inline void load_vector(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, const T* const ptr, const T mul, const bool fill = true) {
	detail_namespace::load_vector(frag, ptr, mul, fill);
}

template <int M, int N, int K, class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const nvcuda::wmma::layout_t layout) {
	detail_namespace::store_vector(ptr, frag, layout);
}

template <int M, int N, int K, class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	detail_namespace::store_vector(ptr, frag, mul, layout);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class Func, class FT>
__device__ inline void foreach(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, Func func) {
	detail_namespace::foreach(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach(Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	Frag_T frag;
	detail_namespace::foreach(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach_v(Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	Frag_T frag;
	detail_namespace::foreach_v(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach_v(const nvcuda::wmma::layout_t layout, Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	Frag_T frag;
	detail_namespace::foreach_v(frag, layout, func);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class Func, class FT>
__device__ inline void load_matrix_with_operation(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, const T* const ptr, unsigned ldm, Func func) {
	detail_namespace::load_matrix_with_operation(frag, ptr, ldm, func);
}

template <int M, int N, int K, class T>
__device__ inline void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag) {
	detail_namespace::make_identity_matrix(frag);
}

template <int M, int N, int K, class T>
__device__ inline void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const T alpha) {
	detail_namespace::add_eye(frag, alpha);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S, class FT>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<T, S, 2>(frag_x, x, dx, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S, class FT>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<T, S, 3>(frag_x, x, dx, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class FT>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const float* const x,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<2>(frag_x, x, fill);
}

template <class MatrixType, int M, int N, int K, class MemMajor, class FT>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const float* const x,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<3>(frag_x, x, fill);
}

// For only CC >= 80
template <class MatrixType, int M, int N, int K, class MemMajor, class T, class FT>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, const T* const ptr, const bool fill = true) {
#if __CUDA_ARCH__ >= 800
	detail_namespace::load_vector_with_rounding(frag, ptr, fill);
#endif
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class FT>
__device__ inline void load_vector_with_rounding(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, const T* const ptr, const T mul, const bool fill = true) {
#if __CUDA_ARCH__ >= 800
	detail_namespace::load_vector_with_rounding(frag, ptr, mul, fill);
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

#endif /* end of include guard */

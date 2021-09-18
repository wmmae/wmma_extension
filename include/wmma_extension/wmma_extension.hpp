#ifndef __WMMA_EXTENSION_HPP__
#define __WMMA_EXTENSION_HPP__
#include <stdio.h>
#include <type_traits>
#include <mma.h>
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

// ------------------------------
// Primitive functions
// ------------------------------
template <class MatrixType, int M, int N, int K, class MemMajor, class Func, class FT>
__device__ inline void foreach(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, Func func) {
	detail_namespace::foreach(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach(Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	typename std::remove_reference<Frag_T>::type frag;
	detail_namespace::foreach(frag, func);
	__syncwarp();
}

template <class Frag_T, class Func>
__device__ inline void foreach(const nvcuda::wmma::layout_t layout, Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	typename std::remove_reference<Frag_T>::type frag;
	detail_namespace::foreach<typename Frag_T::element_type>(frag, layout, func);
	__syncwarp();
}

template <class MatrixType, int M, int N, int K, class MemMajor, class Func, class FT>
__device__ inline void foreach_ij(nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag, Func func) {
	detail_namespace::foreach_ij(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach_ij(Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	typename std::remove_reference<Frag_T>::type frag;
	detail_namespace::foreach_ij(frag, func);
	__syncwarp();
}

template <class Frag_T, class Func>
__device__ inline void foreach_ij(const nvcuda::wmma::layout_t layout, Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	typename std::remove_reference<Frag_T>::type frag;
	detail_namespace::foreach_ij<typename Frag_T::element_type>(frag, layout, func);
	__syncwarp();
}

template <class Frag_T, class Func>
__device__ inline void foreach_v(Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	detail_namespace::foreach_v(frag, func);
	__syncwarp();
}

template <class Frag_T, class Func>
__device__ inline void foreach_v(const nvcuda::wmma::layout_t layout, Func func) {
	// Requirering `frag` as an argument does not look good but it can not be helped because C++ does not support partial template specialization of a templeta function.
	// The `frag` below does not consume registers because of optimization by nvcc.
	// So this implementation is not a problem.
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	detail_namespace::foreach_v(frag, layout, func);
	__syncwarp();
}

// (i, j) to (tid, frag_i)
template <class Frag_T>
__device__ inline void map(
		unsigned tid_list[2],
		unsigned fid_list[2],
		unsigned& list_size,
		const unsigned i,
		const unsigned j
		) {
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	detail_namespace::map(frag, tid_list, fid_list, list_size, i, j);
	__syncwarp();
}

// ------------------------------
// LD/ST functions for vectors
// ------------------------------
template <class Use, int M, int N, int K, class FT, class Layout, class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<Use, M, N, K, FT, Layout>& frag, const T* const ptr, const bool fill = true) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	mtk::wmma::foreach_v<decltype(frag)>(
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				frag.x[frag_index_list[i]] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[mem_index]);
		});
}

template <class Use, int M, int N, int K, class FT, class Layout, class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<Use, M, N, K, FT, Layout>& frag, const T* const ptr, const T mul, const bool fill = true) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	mtk::wmma::foreach_v<decltype(frag)>(
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				frag.x[frag_index_list[i]] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[mem_index] * mul);
		});
}

template <class Use, int M, int N, int K, class FT, class Layout, class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<Use, M, N, K, FT, Layout>& frag, const T* const ptr, const nvcuda::wmma::layout_t layout, const bool fill = true) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	mtk::wmma::foreach_v<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				frag.x[frag_index_list[i]] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[mem_index]);
		});
}

template <class Use, int M, int N, int K, class FT, class Layout, class T>
__device__ inline void load_vector(nvcuda::wmma::fragment<Use, M, N, K, FT, Layout>& frag, const T* const ptr, const nvcuda::wmma::layout_t layout, const T mul, const bool fill = true) {
	if (fill)
		mtk::wmma::fill_zero(frag);
	mtk::wmma::foreach_v<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				frag.x[frag_index_list[i]] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[mem_index] * mul);
		});
}

template <int M, int N, int K, class FT, class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, FT>& frag, const nvcuda::wmma::layout_t layout) {
	mtk::wmma::foreach_v<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				ptr[mem_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<T>::type>(frag.x[frag_index_list[i]]);
		});
}

template <int M, int N, int K, class FT, class T>
__device__ inline void store_vector(T* const ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, FT>& frag, const T mul, const nvcuda::wmma::layout_t layout) {
	mtk::wmma::foreach_v<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				ptr[mem_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<T>::type>(frag.x[frag_index_list[i]]) * mul;
		});
}

// ------------------------------
// Identity matrix making function
// ------------------------------
template <int M, int N, int K, class T, class FT>
__device__ void add_eye(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag, const FT a) {
	detail_namespace::add_eye(frag, a);
	__syncwarp();
}

template <int M, int N, int K, class T>
__device__ void make_identity_matrix(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T>& frag) {
	mtk::wmma::fill_zero(frag);
	mtk::wmma::add_eye(frag, mtk::wmma::detail::common::cast<T>(1.0f));
	__syncwarp();
}

// ------------------------------
// Loading direct product vector functions
// ------------------------------
template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S, class FT>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<T, S, 2>(frag_x, x, dx, fill);
	__syncwarp();
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T, class S, class FT>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const T* const x, const S* const dx,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<T, S, 3>(frag_x, x, dx, fill);
	__syncwarp();
}

template <class MatrixType, int M, int N, int K, class MemMajor, class FT>
__device__ inline void make_direct_product_fragment(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const float* const x,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<2>(frag_x, x, fill);
	__syncwarp();
}

template <class MatrixType, int M, int N, int K, class MemMajor, class FT>
__device__ inline void make_direct_product_fragment_c3(
		nvcuda::wmma::fragment<MatrixType, M, N, K, FT, MemMajor>& frag_x,
		const float* const x,
		const bool fill = true
		) {
	detail_namespace::make_direct_product_fragment<3>(frag_x, x, fill);
	__syncwarp();
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const nvcuda::wmma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const char* name = "") {
	if ((threadIdx.x & 0x1f) == 0) {
		if (name[0] != '\0') {
			printf("%s = \n", name);
		}
	}
	__syncwarp();
	for (unsigned i = 0; i < warpSize; i++) {
		if (i == (threadIdx.x & 0x1f)) {
			for (unsigned j = 0; j < frag.num_elements; j++) {
				const auto v = mtk::wmma::detail::common::cast<float>(frag.x[j]);
				printf("%+.3e ", v);
			}
			printf("\n");
		}
		__syncwarp();
	}
}

} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

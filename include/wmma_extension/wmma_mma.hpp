#ifndef __WMMAE_WMMA_MMA_HPP__
#define __WMMAE_WMMA_MMA_HPP__
#include "wmma_extension.hpp"
#include "detail/m16n8k16.hpp"
#include "detail/m16n8k8.hpp"
#include "detail/m16n8k8_tf32.hpp"
#include "detail/m8n8k4.hpp"

namespace mtk {
namespace wmma {
namespace mma {
// ------------------------------
// primitive functions for mma fragments
// ------------------------------
//
template <class Frag_T, class Func>
__device__ inline void foreach(Func func) {
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	mtk::wmma::mma::foreach(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach(const nvcuda::wmma::layout_t layout, Func func) {
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	mtk::wmma::mma::foreach(frag, layout, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach_ij(Func func) {
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	mtk::wmma::mma::foreach_ij(frag, func);
}

template <class Frag_T, class Func>
__device__ inline void foreach_ij(const nvcuda::wmma::layout_t layout, Func func) {
	typename std::remove_const<typename std::remove_reference<Frag_T>::type>::type frag;
	mtk::wmma::mma::foreach_ij(frag, layout, func);
}
template <class Frag_T, class Func>
__device__ inline void foreach_v(Func func) {
	typename std::remove_reference<Frag_T>::type frag;
	mtk::wmma::mma::foreach_v(frag, func);
	__syncwarp();
}

template <class Frag_T, class Func>
__device__ inline void foreach_v(const nvcuda::wmma::layout_t layout, Func func) {
	typename std::remove_reference<Frag_T>::type frag;
	mtk::wmma::mma::foreach_v(frag, layout, func);
	__syncwarp();
}

// ------------------------------
// LD/ST functions for mma fragments
// ------------------------------
template <class Use, int M, int N, int K, class FT, class Layout, class T>
__device__ inline void load_matrix_sync(mtk::wmma::mma::fragment<Use, M, N, K, FT, Layout>& frag, const T* const ptr, const unsigned ldm, const bool sync = true) {
	// length of leading dimension of the input fragment
	constexpr unsigned old_ldm = mtk::wmma::detail::common::layout_switch<Layout, mtk::wmma::detail::common::get_M<Use, M, N, K>::value, mtk::wmma::detail::common::get_N<Use, M, N, K>::value>::value;
	mtk::wmma::mma::foreach<decltype(frag)>(
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			const unsigned offset = (mem_index / old_ldm) * ldm + mem_index % old_ldm;
			for (unsigned i = 0; i < fragment_index_count; i++) {
				const unsigned frag_index = frag_index_list[i];
				frag.x[frag_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[offset]);
			}
		});
	if (sync)
		__syncwarp();
}

template <int M, int N, int K, class FT, class T>
__device__ inline void load_matrix_sync(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, FT>& frag, const T* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	// length of leading dimension of the input fragment
	using Use = nvcuda::wmma::accumulator;
	const unsigned old_ldm = (layout == nvcuda::wmma::mem_col_major) ? mtk::wmma::detail::common::get_M<Use, M, N, K>::value : mtk::wmma::detail::common::get_N<Use, M, N, K>::value;
	mtk::wmma::mma::foreach<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			const unsigned offset = (mem_index / old_ldm) * ldm + mem_index % old_ldm;
			for (unsigned i = 0; i < fragment_index_count; i++) {
				const unsigned frag_index = frag_index_list[i];
				frag.x[frag_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[offset]);
			}
		});
	if (sync)
		__syncwarp();
}

template <int M, int N, int K, class FT, class T>
__device__ inline void store_matrix_sync(T* const ptr, const mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, FT>& frag, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	// length of leading dimension of the input fragment
	using Use = nvcuda::wmma::accumulator;
	const unsigned old_ldm = (layout == nvcuda::wmma::mem_col_major) ? mtk::wmma::detail::common::get_M<Use, M, N, K>::value : mtk::wmma::detail::common::get_N<Use, M, N, K>::value;
	mtk::wmma::mma::foreach<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			const unsigned offset = (mem_index / old_ldm) * ldm + mem_index % old_ldm;
			for (unsigned i = 0; i < fragment_index_count; i++) {
				const unsigned frag_index = frag_index_list[i];
				ptr[offset] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<T>::type>(frag.x[frag_index]);
			}
		});
	if (sync)
		__syncwarp();
}

// ------------------------------
// LD/ST vector functions for mma fragments
// ------------------------------
template <class Use, int M, int N, int K, class FT, class Layout, class T>
__device__ inline void load_vector(mtk::wmma::mma::fragment<Use, M, N, K, FT, Layout>& frag, const T* const ptr) {
	mtk::wmma::mma::foreach_v<decltype(frag)>(
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++) {
				const unsigned frag_index = frag_index_list[i];
				frag.x[frag_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[mem_index]);
			}
		});
}

template <int M, int N, int K, class FT, class T>
__device__ inline void load_vector(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, FT>& frag, const T* const ptr, const nvcuda::wmma::layout_t layout) {
	mtk::wmma::mma::foreach_v<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++) {
				const unsigned frag_index = frag_index_list[i];
				frag.x[frag_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<FT>::type>(ptr[mem_index]);
			}
		});
}

template <int M, int N, int K, class FT, class T>
__device__ inline void store_vector(T* const ptr, const mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, FT>& frag, const nvcuda::wmma::layout_t layout) {
	mtk::wmma::mma::foreach_v<decltype(frag)>(layout,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++) {
				const unsigned frag_index = frag_index_list[i];
				ptr[mem_index] = mtk::wmma::detail::common::cast<typename mtk::wmma::detail::common::storage_t<T>::type>(frag.x[frag_index]);
			}
		});
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const mtk::wmma::mma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const char* name = "") {
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
		__syncwarp();
	}
}
} // namespace mma
} // namespace wmma
} // namespace mtk
#endif

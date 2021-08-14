#ifndef __WMMAE_HMMA_F32_F32_DETAIL_FUNCTIONS_SIMT_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_FUNCTIONS_SIMT_HPP__
#include "wmma_extension_simt_include.hpp"
#include "policy_simt.hpp"
namespace mtk {
namespace wmma {
namespace tcec {
namespace detail {
// foreach
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::mma_simt::foreach<typename mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::mma_simt::foreach<typename mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

// foreach_ij
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_ij_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::mma_simt::foreach_ij<typename mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::mma_simt::foreach_ij<typename mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

// foreach_v
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_v_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::mma_simt::foreach_v<typename mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::mma_simt::foreach_v<typename mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

// fill zero
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct fill_zero_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag) {
		mtk::wmma::mma_simt::fill_zero(frag);
	}
};

// load_matrix_sync
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct load_matrix_sync_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma_simt::load_matrix_sync(frag, ptr, ldm, layout);
	}
	__device__ void operator()(mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const unsigned ldm) {
		mtk::wmma::mma_simt::load_matrix_sync(frag, ptr, ldm);
	}
};

// store_matrix_sync
template <class T, class ErrorCorrection, int fm, int fn, int fk>
struct store_matrix_sync_wrapper<T, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(float* ptr, mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, fm, fn, fk, float>& frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma_simt::store_matrix_sync(ptr, frag, ldm, layout);
	}
};

// load_vector
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct load_vector_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma_simt::load_vector(frag, ptr, layout);
	}
	__device__ void operator()(mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr) {
		mtk::wmma::mma_simt::load_vector(frag, ptr);
	}
};

// store_vector
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct store_vector_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(float* ptr, mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma_simt::store_vector(ptr, frag, layout);
	}
};

// fill_fragment
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk, class VT>
struct fill_fragment_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>, VT> {
	__device__ void operator()(mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>& frag, const VT v) {
		mtk::wmma::mma_simt::fill_fragment(frag, v);
	}
};

// mma_sync
template <class AB_T, class A_Layout, class B_Layout, class CD_T, class ErrorCorrection, int fm, int fn, int fk>
struct mma_sync_wrapper<AB_T, A_Layout, B_Layout, CD_T, Policy<mtk::wmma::tcec::op_simt, ErrorCorrection, fm, fn, fk>> {
	using Fragment_A = mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_a, fm, fn, fk, AB_T, A_Layout>;
	using Fragment_B = mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_b, fm, fn, fk, AB_T, B_Layout>;
	using Fragment_C = mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, fm, fn, fk, CD_T>;
	__device__ void operator()(Fragment_C& d, const Fragment_A& a, const Fragment_B& b, const Fragment_C& c) {
		mtk::wmma::mma_simt::mma_sync(d, a, b, c);
	}
};

} // namespace detail
} // namespace tcec
} // namespace wmma
} // namespace mtk
#endif

#ifndef __WMMAE_HMMA_F32_F32_DETAIL_FUNCTIONS_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_FUNCTIONS_HPP__
#include "wmma_extension_include.hpp"
#include "policy.hpp"
namespace mtk {
namespace wmma {
namespace tcec {
namespace detail {
// foreach
template <class Use, class T, class Layout, class Policy>
struct foreach_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::foreach<typename nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::foreach<typename nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::mma::foreach<typename mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::mma::foreach<typename mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

// foreach_ij
template <class Use, class T, class Layout, class Policy>
struct foreach_ij_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_ij_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::foreach_ij<typename nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::foreach_ij<typename nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_ij_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::mma::foreach_ij<typename mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::mma::foreach_ij<typename mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

// foreach_v
template <class Use, class T, class Layout, class Policy>
struct foreach_v_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_v_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::foreach_v<typename nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::foreach_v<typename nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct foreach_v_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	template <class Func>
	__device__ void operator()(Func func) {
		mtk::wmma::mma::foreach_v<typename mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>>(
				func
				);
	}
	template <class Func>
	__device__ void operator()(const nvcuda::wmma::layout_t layout, Func func) {
		mtk::wmma::mma::foreach_v<typename mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>>(
				layout, func
				);
	}
};

// fill zero
template <class Use, class T, class Layout, class Policy>
struct fill_zero_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct fill_zero_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag) {
		mtk::wmma::fill_zero(frag);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct fill_zero_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag) {
		mtk::wmma::mma::fill_zero(frag);
	}
};

// load_matrix_sync
template <class Use, class T, class Layout, class Policy>
struct load_matrix_sync_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct load_matrix_sync_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, unsigned ldm, const nvcuda::wmma::layout_t layout) {
		nvcuda::wmma::load_matrix_sync(frag, ptr, ldm, layout);
	}
	__device__ void operator()(nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, unsigned ldm) {
		nvcuda::wmma::load_matrix_sync(frag, ptr, ldm);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct load_matrix_sync_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::load_matrix_sync(frag, ptr, ldm, layout);
	}
	__device__ void operator()(mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const unsigned ldm) {
		mtk::wmma::mma::load_matrix_sync(frag, ptr, ldm);
	}
};

// store_matrix_sync
template <class T, class Policy>
struct store_matrix_sync_wrapper;

template <class T, class ErrorCorrection, int fm, int fn, int fk>
struct store_matrix_sync_wrapper<T, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(float* ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, fm, fn, fk, float>& frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		nvcuda::wmma::store_matrix_sync(ptr, frag, ldm, layout);
	}
};

template <class T, class ErrorCorrection, int fm, int fn, int fk>
struct store_matrix_sync_wrapper<T, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(float* ptr, mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, fm, fn, fk, float>& frag, const unsigned ldm, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::store_matrix_sync(ptr, frag, ldm, layout);
	}
};

// load_vector
template <class Use, class T, class Layout, class Policy>
struct load_vector_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct load_vector_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::load_vector(frag, ptr, layout);
	}
	__device__ void operator()(nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr) {
		mtk::wmma::load_vector(frag, ptr);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct load_vector_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::load_vector(frag, ptr, layout);
	}
	__device__ void operator()(mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag, const float* const ptr) {
		mtk::wmma::mma::load_vector(frag, ptr);
	}
};

// store_vector
template <class Use, class T, class Layout, class Policy>
struct store_vector_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct store_vector_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(float* ptr, nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::store_vector(ptr, frag, layout);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct store_vector_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	__device__ void operator()(float* ptr, mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag, const nvcuda::wmma::layout_t layout) {
		mtk::wmma::mma::store_vector(ptr, frag, layout);
	}
};

// fill_fragment
template <class Use, class T, class Layout, class Policy, class VT>
struct fill_fragment_wrapper;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk, class VT>
struct fill_fragment_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>, VT> {
	__device__ void operator()(nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>& frag, const VT v) {
		nvcuda::wmma::fill_fragment(frag, v);
	}
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk, class VT>
struct fill_fragment_wrapper<Use, T, Layout, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>, VT> {
	__device__ void operator()(mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>& frag, const VT v) {
		mtk::wmma::mma::fill_fragment(frag, v);
	}
};

// mma_sync
template <class AB_T, class A_Layout, class B_Layout, class CD_T, class Policy>
struct mma_sync_wrapper;

template <class AB_T, class A_Layout, class B_Layout, class CD_T, class ErrorCorrection, int fm, int fn, int fk>
struct mma_sync_wrapper<AB_T, A_Layout, B_Layout, CD_T, Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, fm, fn, fk>> {
	using Fragment_A = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, fm, fn, fk, AB_T, A_Layout>;
	using Fragment_B = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, fm, fn, fk, AB_T, B_Layout>;
	using Fragment_C = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, fm, fn, fk, CD_T>;
	__device__ void operator()(Fragment_C& d, const Fragment_A& a, const Fragment_B& b, const Fragment_C& c) {
		nvcuda::wmma::mma_sync(d, a, b, c);
	}
};

template <class AB_T, class A_Layout, class B_Layout, class CD_T, class ErrorCorrection, int fm, int fn, int fk>
struct mma_sync_wrapper<AB_T, A_Layout, B_Layout, CD_T, Policy<mtk::wmma::tcec::op_mma, ErrorCorrection, fm, fn, fk>> {
	using Fragment_A = mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, fm, fn, fk, AB_T, A_Layout>;
	using Fragment_B = mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, fm, fn, fk, AB_T, B_Layout>;
	using Fragment_C = mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, fm, fn, fk, CD_T>;
	__device__ void operator()(Fragment_C& d, const Fragment_A& a, const Fragment_B& b, const Fragment_C& c) {
		mtk::wmma::mma::mma_sync(d, a, b, c);
	}
};

} // namespace detail
} // namespace tcec
} // namespace wmma
} // namespace mtk
#endif

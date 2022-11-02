#ifndef __WMMAE_TCEC_COMPLEX_HPP__
#define __WMMAE_TCEC_COMPLEX_HPP__
#include <cuComplex.h>
#include "tcec.hpp"

namespace mtk {
namespace wmma {
namespace tcec {
template <class Use, int m, int n, int k, class T, class Layout = void,
		 class Policy_ = typename mtk::wmma::tcec::detail::default_policy<T>::type>
struct fragment_complex {
	using frag_t = mtk::wmma::tcec::fragment<Use, m, n, k, T, Layout, Policy_>;
	static constexpr unsigned num_sub_frag_m = frag_t::num_sub_frag_m;
	static constexpr unsigned num_sub_frag_n = frag_t::num_sub_frag_n;
	frag_t real, imag;
};

template <class Use, int m, int n, int k, class T, class Layout, class Policy>
__device__ void fill_fragment(fragment_complex<Use, m, n, k, T, Layout, Policy>& frag,
		const typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type>::type real,
		const typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type>::type imag = 0
		) {
	mtk::wmma::tcec::fill_fragment(frag.real, real);
	mtk::wmma::tcec::fill_fragment(frag.imag, imag);
}

template <class Use, int m, int n, int k, class T, class Layout, class Policy>
__device__ void fill_zero(fragment_complex<Use, m, n, k, T, Layout, Policy>& frag
		) {
	mtk::wmma::tcec::fill_zero(frag.real);
	mtk::wmma::tcec::fill_zero(frag.imag);
}

// Load
template <class Layout, int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, Layout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto frag_index = frag_index_list[0];
						const auto v = ptr[mem_offset];
						frag.real.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = v.x;
						frag.imag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = v.y;
						frag.real.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = 0.f;
						frag.imag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = 0.f;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Layout, int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, Layout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto frag_index = frag_index_list[0];
						const auto v = ptr[mem_offset];
						frag.real.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = v.x;
						frag.imag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = v.y;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <int m, int n, int k, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	if (layout == nvcuda::wmma::mem_col_major) {
		load_matrix_sync<nvcuda::wmma::col_major>(frag, ptr, ldm, sync);
	} else {
		load_matrix_sync<nvcuda::wmma::row_major>(frag, ptr, ldm, sync);
	}
}

// -----------
// load_matrix_sync
// -----------
template <class MatrixLayout, class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto cv = ptr[mem_offset];
						{
							const auto v = cv.x;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.real.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
								frag.real.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
							}
						}
						{
							const auto v = cv.y;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.imag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
								frag.imag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
							}
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class MatrixLayout, class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto cv = ptr[mem_offset];
						{
							const auto v = cv.x;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.real.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							}
						}
						{
							const auto v = cv.y;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.imag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							}
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, class Ec, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag, const float* const ptr, const unsigned ldm, const bool sync = true) {
	load_matrix_sync<Layout>(frag, ptr, ldm, sync);
}

// -----------
// load_matrix_sync_with_mul
// -----------
template <class MatrixLayout, class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync_with_mul(fragment_complex<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const float mul, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto cv = ptr[mem_offset];
						{
							const auto v = cv.x * mul;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.real.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
								frag.real.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
							}
						}
						{
							const auto v = cv.y * mul;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							const auto dhv = mtk::wmma::detail::common::cast<T>(detail::correction_scale_0<T>(v - mtk::wmma::detail::common::cast<float>(hv)));
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.imag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
								frag.imag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = dhv;
							}
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class MatrixLayout, class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
__device__ void load_matrix_sync(fragment_complex<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const cuComplex* const ptr, const unsigned ldm, const float mul, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto cv = ptr[mem_offset];
						{
							const auto v = cv.x * mul;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.real.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							}
						}
						{
							const auto v = cv.y * mul;
							const auto hv = mtk::wmma::detail::common::cast<T>(v);
							for (unsigned f = 0; f < frag_index_count; f++) {
								const auto frag_index = frag_index_list[f];
								frag.imag.sub_frag  [bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
							}
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, class Ec, int fm, int fn, int fk>
__device__ void load_matrix_sync_with_mul(fragment_complex<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag, const float* const ptr, const unsigned ldm, const float mul, const bool sync = true) {
	load_matrix_sync_with_mul<Layout>(frag, ptr, ldm, mul, sync);
}

// -----------
// store_matrix_sync
// -----------
template <class Layout, int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_matrix_sync(cuComplex* const ptr, fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>& frag, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, Layout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto frag_index = frag_index_list[0];
						cuComplex c;
						c.x = (frag.real.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] + detail::correction_scale_1<T>(frag.real.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index]));
						c.y = (frag.imag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] + detail::correction_scale_1<T>(frag.imag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index]));
						ptr[mem_offset] = c;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Layout, int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_matrix_sync(cuComplex* const ptr, fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, Layout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto frag_index = frag_index_list[0];
						cuComplex c;
						c.x = frag.real.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index];
						c.y = frag.imag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index];
						ptr[mem_offset] = c;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <int m, int n, int k, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void store_matrix_sync(cuComplex* const ptr, fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	if (layout == nvcuda::wmma::mem_col_major) {
		store_matrix_sync<nvcuda::wmma::col_major>(ptr, frag, ldm, sync);
	} else {
		store_matrix_sync<nvcuda::wmma::row_major>(ptr, frag, ldm, sync);
	}
}

// -----------
// store_matrix_sync_with_mul
// -----------
template <class Layout, int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_matrix_sync(cuComplex* const ptr, fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>& frag, const unsigned ldm, const float mul, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, Layout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto frag_index = frag_index_list[0];
						cuComplex c;
						c.x = (frag.real.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] + detail::correction_scale_1<T>(frag.real.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index]));
						c.y = (frag.imag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] + detail::correction_scale_1<T>(frag.imag.sub_d_frag[bm + frag.num_sub_frag_m * bn].x[frag_index]));
						ptr[mem_offset] = c;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Layout, int m, int n, int k, class T, class Op, int fm, int fn, int fk>
__device__ void store_matrix_sync(cuComplex* const ptr, fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const unsigned ldm, const float mul, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, Layout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto frag_index = frag_index_list[0];
						cuComplex c;
						c.x = frag.real.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index];
						c.y = frag.imag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index];
						ptr[mem_offset] = c;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <int m, int n, int k, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void store_matrix_sync(cuComplex* const ptr, fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag, const unsigned ldm, const float mul, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	if (layout == nvcuda::wmma::mem_col_major) {
		store_matrix_sync<nvcuda::wmma::col_major>(ptr, frag, ldm, mul, sync);
	} else {
		store_matrix_sync<nvcuda::wmma::row_major>(ptr, frag, ldm, mul, sync);
	}
}

// -----------
// mma
// -----------
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void mma_sync(
		fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_d,
		const fragment_complex<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_a,
		const fragment_complex<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_b,
		const fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_c) {
	mtk::wmma::tcec::mma_sync(frag_d.real, frag_a.real, frag_b.real, frag_c.real);
	mtk::wmma::tcec::mma_sync(frag_d.imag, frag_a.imag, frag_b.real, frag_c.imag);
	mtk::wmma::tcec::mma_sync(frag_d.imag, frag_a.real, frag_b.imag, frag_d.imag);
	mtk::wmma::tcec::mma_sync(frag_d.real, frag_a.imag, mtk::wmma::tcec::neg(frag_b.imag), frag_d.real);
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void mma_sync(
		fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_d,
		const fragment_complex<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_a,
		const fragment_complex<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_b) {
	mtk::wmma::tcec::mma_sync(frag_d.imag, frag_a.imag, frag_b.real);
	mtk::wmma::tcec::mma_sync(frag_d.imag, frag_a.real, frag_b.imag, frag_d.imag);
	mtk::wmma::tcec::mma_sync(frag_d.real, frag_a.real, frag_b.real);
	mtk::wmma::tcec::mma_sync(frag_d.real, frag_a.imag, mtk::wmma::tcec::neg(frag_b.imag), frag_d.real);
}

// -----------
// mma_rz
// -----------
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void mma_rz_sync(
		fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_d,
		const fragment_complex<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_a,
		const fragment_complex<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_b,
		const fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_c) {
	mtk::wmma::tcec::mma_rz_sync(frag_d.real, frag_a.real, frag_b.real, frag_c.real);
	mtk::wmma::tcec::mma_rz_sync(frag_d.imag, frag_a.imag, frag_b.real, frag_c.imag);
	mtk::wmma::tcec::mma_rz_sync(frag_d.imag, frag_a.real, frag_b.imag, frag_d.imag);
	mtk::wmma::tcec::mma_rz_sync(frag_d.real, frag_a.imag, mtk::wmma::tcec::neg(frag_b.imag), frag_d.real);
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void mma_rz_sync(
		fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_d,
		const fragment_complex<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_a,
		const fragment_complex<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_b) {
	mtk::wmma::tcec::mma_rz_sync(frag_d.imag, frag_a.imag, frag_b.real);
	mtk::wmma::tcec::mma_rz_sync(frag_d.imag, frag_a.real, frag_b.imag, frag_d.imag);
	mtk::wmma::tcec::mma_rz_sync(frag_d.real, frag_a.real, frag_b.real);
	mtk::wmma::tcec::mma_rz_sync(frag_d.real, frag_a.imag, mtk::wmma::tcec::neg(frag_b.imag), frag_d.real);
}

// -----------
// mma_rn
// -----------
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void mma_rn_sync(
		fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_d,
		const fragment_complex<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_a,
		const fragment_complex<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_b,
		const fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_c) {
	mtk::wmma::tcec::mma_rn_sync(frag_d.real, frag_a.real, frag_b.real, frag_c.real);
	mtk::wmma::tcec::mma_rn_sync(frag_d.imag, frag_a.imag, frag_b.real, frag_c.imag);
	mtk::wmma::tcec::mma_rn_sync(frag_d.imag, frag_a.real, frag_b.imag, frag_d.imag);
	mtk::wmma::tcec::mma_rn_sync(frag_d.real, frag_a.imag, mtk::wmma::tcec::neg(frag_b.imag), frag_d.real);
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, class Ec, int fm, int fn, int fk>
__device__ void mma_rn_sync(
		fragment_complex<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_d,
		const fragment_complex<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_a,
		const fragment_complex<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, Ec, fm, fn, fk>>& frag_b) {
	mtk::wmma::tcec::mma_rn_sync(frag_d.imag, frag_a.imag, frag_b.real);
	mtk::wmma::tcec::mma_rn_sync(frag_d.imag, frag_a.real, frag_b.imag, frag_d.imag);
	mtk::wmma::tcec::mma_rn_sync(frag_d.real, frag_a.real, frag_b.real);
	mtk::wmma::tcec::mma_rn_sync(frag_d.real, frag_a.imag, mtk::wmma::tcec::neg(frag_b.imag), frag_d.real);
}

} // namespace tcec
} // namespace wmma
} // namespace mtk
#endif

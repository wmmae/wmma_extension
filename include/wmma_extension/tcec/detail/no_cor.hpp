#ifndef __MTK_HMMA_F32_F32_NO_COR_HPP__
#define __MTK_HMMA_F32_F32_NO_COR_HPP__

#include <type_traits>

#include "common.hpp"
#include "policy.hpp"
#include "functions.hpp"

namespace mtk {
namespace wmma {
namespace tcec {
template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk>
struct fragment <Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> {
	using element_type = float;
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	static const int sub_frag_m = Policy::m;
	static const int sub_frag_n = Policy::n;
	static const int sub_frag_k = Policy::k;

	using sub_frag_t = typename mtk::wmma::tcec::detail::default_fragment<Use, typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type, Layout, Policy>::type;
	static constexpr int num_sub_frag_m = mtk::wmma::tcec::detail::select_value<Use, m, k, m>::value / mtk::wmma::tcec::detail::select_value<Use, sub_frag_m, sub_frag_k, sub_frag_m>::value;
	static constexpr int num_sub_frag_n = mtk::wmma::tcec::detail::select_value<Use, k, n, n>::value / mtk::wmma::tcec::detail::select_value<Use, sub_frag_k, sub_frag_n, sub_frag_n>::value;

	sub_frag_t sub_frag  [num_sub_frag_m * num_sub_frag_n];
	sub_frag_t sub_d_frag[0];

	static const unsigned num_elements = num_sub_frag_m * num_sub_frag_n * sub_frag_t::num_elements;
	__device__ typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type>::type& x(const unsigned index) {
		const auto frag_index = index % sub_frag_t::num_elements;
		const auto sub_frag_id = index / sub_frag_t::num_elements;
		return sub_frag[sub_frag_id].x[frag_index];
	}

	// const version
	__device__ typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type>::type x(const unsigned index) const {
		const auto frag_index = index % sub_frag_t::num_elements;
		const auto sub_frag_id = index / sub_frag_t::num_elements;
		return sub_frag[sub_frag_id].x[frag_index];
	}

	// integrate
	__device__ void integrate() {}
};

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> neg(
		const fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag
		) {
	fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> res;
	for (unsigned i = 0; i < res.num_elements; i++) {
		res.x(i)  = -frag.x(i);
	}
	return res;
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void fill_fragment(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type>::type v) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	using sub_frag_t = typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type;
	using storage_t = typename mtk::wmma::detail::common::storage_t<typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type>::type;
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::tcec::detail::fill_fragment_wrapper<Use, sub_frag_t, Layout, Policy, storage_t>{}(frag.sub_frag  [bm + frag.num_sub_frag_m * bn], v);
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void fill_zero(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag) {
	using sub_frag_t = typename mtk::wmma::tcec::detail::sub_frag_t<Use, T>::type;
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::tcec::detail::fill_zero_wrapper<Use, sub_frag_t, Layout, Policy>{}(frag.sub_frag  [bm + frag.num_sub_frag_m * bn]);
		}
	}
}

// Load matrix
template <class MatrixLayout, int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_matrix_sync(fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(layout,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset];
						const auto hv = mtk::wmma::detail::common::cast<T>(v);
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_matrix_sync(fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	if (layout == nvcuda::wmma::mem_col_major) {
		load_matrix_sync<nvcuda::wmma::col_major>(frag, ptr, ldm, layout, sync);
	} else {
		load_matrix_sync<nvcuda::wmma::row_major>(frag, ptr, ldm, layout, sync);
	}
}

template <class MatrixLayout, class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_matrix_sync(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset];
						const auto hv = mtk::wmma::detail::common::cast<T>(v);
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_matrix_sync(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const unsigned ldm, const bool sync = true) {
	load_matrix_sync<Layout>(frag, ptr, ldm, sync);
}

template <class MatrixLayout, class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_matrix_sync_with_mul(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const unsigned ldm, const MEM_T mul, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						const auto v = ptr[mem_offset] * mul;
						const auto hv = mtk::wmma::detail::common::cast<T>(v);
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index] = hv ;
						}
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_matrix_sync(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const unsigned ldm, const MEM_T mul, const bool sync = true) {
	load_matrix_sync<Layout>(frag, ptr, ldm, mul, sync);
}

// Store matrix
template <class MatrixLayout, int m, int n, int k, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void store_matrix_sync(float* const ptr, fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> frag, const unsigned ldm, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<MatrixLayout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						ptr[mem_offset] = frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index_list[0]];
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void store_matrix_sync(MEM_T* const ptr, fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> frag, const unsigned ldm, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	if (layout == nvcuda::wmma::mem_col_major) {
		store_matrix_sync<nvcuda::wmma::col_major>(ptr, frag, ldm, sync);
	} else {
		store_matrix_sync<nvcuda::wmma::row_major>(ptr, frag, ldm, sync);
	}
}

template <class MatrixLayout, int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void store_matrix_sync_with_mul(MEM_T* const ptr,
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> frag,
		const unsigned ldm, const MEM_T mul, const bool sync = true) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	mtk::wmma::tcec::detail::foreach_ij_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(std::is_same<MatrixLayout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned i, const unsigned j) {
				for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						const auto mem_offset = mtk::wmma::tcec::detail::compute_mem_offset<frag_m, frag_n, MatrixLayout>{}(i, j, ldm, bm * frag_m, bn * frag_n);
						ptr[mem_offset] = frag.sub_frag[bm + frag.num_sub_frag_m * bn].x[frag_index_list[0]] * mul;
					}
				}
			});
	if (sync) {
		__syncwarp();
	}
}

template <int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void store_matrix_sync_with_mul(MEM_T* const ptr,
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>> frag,
		const unsigned ldm, const MEM_T mul, const nvcuda::wmma::layout_t layout, const bool sync = true) {
	if (layout == nvcuda::wmma::mem_col_major) {
		store_matrix_sync<nvcuda::wmma::col_major>(ptr, frag, ldm, mul, sync);
	} else {                                                
		store_matrix_sync<nvcuda::wmma::row_major>(ptr, frag, ldm, mul, sync);
	}
}

// Load vector
template <int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_vector(fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	if (layout == nvcuda::wmma::mem_col_major) {
		for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
			mtk::wmma::tcec::detail::load_vector_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_frag[bm], ptr + bm * frag_m, layout);
		}
	} else {
		for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
			mtk::wmma::tcec::detail::load_vector_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(frag.sub_frag[bn * frag.num_sub_frag_m], ptr + bn * frag_n, layout);
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_vector(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	constexpr auto num_load_blocks = mtk::wmma::tcec::detail::layout_switch<Layout, frag.num_sub_frag_m, frag.num_sub_frag_n>::value;
	constexpr auto block_ld        = mtk::wmma::tcec::detail::layout_switch<Layout, 1, frag.num_sub_frag_m>::value;
	constexpr auto vec_per_block   = mtk::wmma::tcec::detail::layout_switch<Layout, frag_m, frag_n>::value;

	mtk::wmma::tcec::detail::foreach_v_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bn = 0; bn < num_load_blocks; bn++) {
					const auto mem_offset = mem_index + bn * vec_per_block;
					const auto v = ptr[mem_offset];
					const auto hv = mtk::wmma::detail::common::cast<T>(v);
					for (unsigned i = 0; i < frag_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag.sub_frag  [bn * block_ld].x[frag_index] = hv ;
					}
				}
			});
}

template <class Use, int m, int n, int k, class T, class Layout, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void load_vector(fragment<Use, m, n, k, T, Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const MEM_T* const ptr, const float mul) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<Use, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<Use, Policy::k, Policy::n, Policy::n>::value;

	constexpr auto num_load_blocks = mtk::wmma::tcec::detail::layout_switch<Layout, frag.num_sub_frag_m, frag.num_sub_frag_n>::value;
	constexpr auto block_ld        = mtk::wmma::tcec::detail::layout_switch<Layout, 1, frag.num_sub_frag_m>::value;
	constexpr auto vec_per_block   = mtk::wmma::tcec::detail::layout_switch<Layout, frag_m, frag_n>::value;

	mtk::wmma::tcec::detail::foreach_v_wrapper<Use, T, Layout, Policy>{}(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				for (unsigned bn = 0; bn < num_load_blocks; bn++) {
					const auto mem_offset = mem_index + bn * vec_per_block;
					const auto v = ptr[mem_offset] * mul;
					const auto hv = mtk::wmma::detail::common::cast<T>(v);
					for (unsigned i = 0; i < frag_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag.sub_frag  [bn * block_ld].x[frag_index] = hv ;
					}
				}
			});
}

// Store vector
template <int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void store_vector(MEM_T* const ptr, fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	if (layout == nvcuda::wmma::mem_col_major) {
		mtk::wmma::tcec::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bm].x[frag_index];
							ptr[bm * frag_m + mem_index] = hv;
						}
					}
				});
	} else {
		mtk::wmma::tcec::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bn * frag.num_sub_frag_m].x[frag_index];
							ptr[bn * frag_n + mem_index] = hv;
						}
					}
				});
	}
}

template <int m, int n, int k, class T, class Op, int fm, int fn, int fk, class MEM_T,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void store_vector(MEM_T* const ptr, fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag, const float mul, const nvcuda::wmma::layout_t layout) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr auto frag_m = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::m, Policy::k, Policy::m>::value;
	constexpr auto frag_n = mtk::wmma::tcec::detail::select_value<nvcuda::wmma::accumulator, Policy::k, Policy::n, Policy::n>::value;

	if (layout == nvcuda::wmma::mem_col_major) {
		mtk::wmma::tcec::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bm = 0; bm < frag.num_sub_frag_m; bm++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bm].x[frag_index];
							ptr[bm * frag_m + mem_index] = hv * mul;
						}
					}
				});
	} else {
		mtk::wmma::tcec::detail::foreach_v_wrapper<nvcuda::wmma::accumulator, float, void, Policy>{}(
				layout,
				[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned bn = 0; bn < frag.num_sub_frag_n; bn++) {
						for (unsigned i = 0; i < frag_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							const auto hv  = frag.sub_frag  [bn * frag.num_sub_frag_m].x[frag_index];
							ptr[bn * frag_n + mem_index] = hv * mul;
						}
					}
				});
	}
}

// mma_rz
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void mma_rz_sync(
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_d,
		const fragment<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_a,
		const fragment<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_b,
		const fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_c) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::tcec::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, Policy> mma_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			mma_op(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_c.sub_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				mma_op(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void mma_rz_sync(
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_d,
		const fragment<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_a,
		const fragment<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_b) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::tcec::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, Policy> mma_op;
	mtk::wmma::tcec::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, Policy> zero_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			zero_op(frag_d.sub_frag[bm + bn * num_m_block]);
			mma_op(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_d.sub_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				mma_op(
						frag_d.sub_frag[bm + bn * num_m_block],
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						frag_d.sub_frag[bm + bn * num_m_block]
						);
			}
		}
	}
}

// mma_rn
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void mma_rn_sync(
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_d,
		const fragment<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_a,
		const fragment<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_b,
		const fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_c) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::tcec::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, Policy> mma_op;
	mtk::wmma::tcec::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, Policy> zero_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			typename fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>::sub_frag_t tmp;
			zero_op(tmp);
			mma_op(
					tmp,
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					tmp
					);
			for (unsigned i = 0; i < tmp.num_elements; i++) {
				frag_d.sub_frag[bm + bn * num_m_block].x[i] = frag_c.sub_frag[bm + bn * num_m_block].x[i] + tmp.x[i];
			}
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				zero_op(tmp);
				mma_op(
						tmp,
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						tmp
						);
				for (unsigned i = 0; i < tmp.num_elements; i++) {
					frag_d.sub_frag[bm + bn * num_m_block].x[i] += tmp.x[i];
				}
			}
		}
	}
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void mma_rn_sync(
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_d,
		const fragment<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_a,
		const fragment<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_b) {
	using Policy = mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>;
	constexpr unsigned num_m_block = frag_d.num_sub_frag_m;
	constexpr unsigned num_n_block = frag_d.num_sub_frag_n;
	constexpr unsigned num_k_block = frag_a.num_sub_frag_n;

	mtk::wmma::tcec::detail::mma_sync_wrapper<T, A_Layout, B_Layout, float, Policy> mma_op;
	mtk::wmma::tcec::detail::fill_zero_wrapper<nvcuda::wmma::accumulator, float, void, Policy> zero_op;

	for (unsigned bm = 0; bm < num_m_block; bm++) {
		for (unsigned bn = 0; bn < num_n_block; bn++) {
			typename fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::with_ec, fm, fn, fk>>::sub_frag_t tmp;
			zero_op(frag_d.sub_frag[bm + bn * num_m_block]);
			mma_op(
					frag_d.sub_frag[bm + bn * num_m_block],
					frag_a.sub_frag[bm + 0  * num_m_block],
					frag_b.sub_frag[0  + bn * num_k_block],
					frag_d.sub_frag[bm + bn * num_m_block]
					);
			for (unsigned bk = 1; bk < num_k_block; bk++) {
				zero_op(tmp);
				mma_op(
						tmp,
						frag_a.sub_frag[bm + bk * num_m_block],
						frag_b.sub_frag[bk + bn * num_k_block],
						tmp
						);
				for (unsigned i = 0; i < tmp.num_elements; i++) {
					frag_d.sub_frag[bm + bn * num_m_block].x[i] += tmp.x[i];
				}
			}
		}
	}
}

// mma
template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void mma_sync(
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_d,
		const fragment<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_a,
		const fragment<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_b,
		const fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_c) {
	mma_rz_sync(frag_d, frag_a, frag_b, frag_c);
}

template <int m, int n, int k, class A_Layout, class B_Layout, class T, class Op, int fm, int fn, int fk,
				 typename std::enable_if<(std::is_same<Op, mtk::wmma::tcec::op_mma>::value || std::is_same<Op, mtk::wmma::tcec::op_wmma>::value), bool>::type = false>
__device__ void mma_sync(
		fragment<nvcuda::wmma::accumulator, m, n, k, T, void, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_d,
		const fragment<nvcuda::wmma::matrix_a, m, n, k, T, A_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_a,
		const fragment<nvcuda::wmma::matrix_b, m, n, k, T, B_Layout, mtk::wmma::tcec::Policy<Op, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag_b) {
	mma_rz_sync(frag_d, frag_a, frag_b);
}
} // namespace tcec
} // namespace wmma
} // namespace mtk
#endif

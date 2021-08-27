#ifndef __WMMAE_TEST_COMMON_HPP__
#define __WMMAE_TEST_COMMON_HPP__
#include <string>
#include <wmma_extension/wmma_extension.hpp>

namespace mtk {
namespace test_utils {
template <class T>
std::string get_string();
template <> std::string get_string<float>() {return "float";}
template <> std::string get_string<half >() {return "half";}
template <> std::string get_string<nvcuda::wmma::precision::tf32>() {return "tf32";}
template <> std::string get_string<nvcuda::wmma::col_major>() {return "col_major";}
template <> std::string get_string<nvcuda::wmma::row_major>() {return "row_major";}
template <> std::string get_string<void>() {return "void";}
template <> std::string get_string<nvcuda::wmma::matrix_a>()  {return "matrix_a";}
template <> std::string get_string<nvcuda::wmma::matrix_b>()  {return "matrix_b";}
template <> std::string get_string<nvcuda::wmma::accumulator>()  {return "accumulator";}

template <class T>
double get_machine_eps();
template <>
double get_machine_eps<half >(){return 1./(1 << 10);}
template <>
double get_machine_eps<nvcuda::wmma::precision::tf32>(){return 1./(1 << 10);}
template <>
double get_machine_eps<float>(){return 1./(1 << 23);}

const char* get_test_result_string(const bool passed) {
	return passed ? "PASSED" : "FAILED";
}

template <int M, int N, int K, class AB_T, class C_T, class D_T, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
double get_max_relative_error(
		const typename mtk::wmma::detail::common::storage_t<AB_T>::type* const a,
		const typename mtk::wmma::detail::common::storage_t<AB_T>::type* const b,
		const C_T* const c,
		const D_T* const d
		) {
	double max_base = 0.0;
	double max_diff = 0.0;

	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double c_v = 0.0;
			for (unsigned k = 0; k < K; k++) {
				double a_v, b_v;
				if (std::is_same<a_layout, nvcuda::wmma::col_major>::value) {
					a_v = mtk::wmma::detail::common::cast<float>(a[k * M + m]);
				} else {
					a_v = mtk::wmma::detail::common::cast<float>(a[k + K * m]);
				}
				if (std::is_same<b_layout, nvcuda::wmma::col_major>::value) {
					b_v = mtk::wmma::detail::common::cast<float>(b[k + K * n]);
				} else {
					b_v = mtk::wmma::detail::common::cast<float>(b[k * N + n]);
				}
				c_v += a_v * b_v;
			}
			if (c_layout == nvcuda::wmma::mem_col_major) {
				c_v += mtk::wmma::detail::common::cast<float>(c[m + M * n]);
			} else {
				c_v += mtk::wmma::detail::common::cast<float>(c[m * N + n]);
			}

			// compute error
			double d_v;
			if (d_layout == nvcuda::wmma::mem_col_major) {
				d_v = mtk::wmma::detail::common::cast<float>(d[m + M * n]);
			} else {
				d_v = mtk::wmma::detail::common::cast<float>(d[m * N + n]);
			}
			const auto diff = d_v - c_v;

			// accumulate
			max_base = std::max(max_base, std::abs(c_v));
			max_diff = std::max(max_diff, std::abs(diff));
		}
	}
	return max_diff / max_base;
}
} // namespace test_utils
} // namespace mtk
#endif

#include <iostream>
#include <type_traits>
#include <random>
#include <mma.h>
#include <wmma_extension/wmma_mma.hpp>
#include "common.hpp"

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

template <int M, int N, int K, class AB_T, class C_T, class D_T, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
__global__ void test_kernel(
		D_T* const d,
		const typename mtk::wmma::detail::common::storage_t<AB_T>::type* const a,
		const typename mtk::wmma::detail::common::storage_t<AB_T>::type* const b,
		const C_T* const c) {
	using AB_STORAGE_T = typename mtk::wmma::detail::common::storage_t<AB_T>::type;
	mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a   , M, N, K, AB_T, a_layout> frag_a;
	mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b   , M, N, K, AB_T, b_layout> frag_b;
	mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, C_T> frag_c;
	mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, D_T> frag_d;

	const unsigned lda = std::is_same<a_layout, nvcuda::wmma::col_major>::value ? M : K;
	const unsigned ldb = std::is_same<b_layout, nvcuda::wmma::col_major>::value ? K : N;
	const unsigned ldc = (c_layout == nvcuda::wmma::mem_col_major) ? M : N;
	const unsigned ldd = (d_layout == nvcuda::wmma::mem_col_major) ? M : N;

	mtk::wmma::mma::fill_zero(frag_d);

	mtk::wmma::mma::load_matrix_sync(frag_a, a, lda);
	mtk::wmma::mma::load_matrix_sync(frag_b, b, ldb);
	mtk::wmma::mma::load_matrix_sync(frag_c, c, ldc, c_layout);

	mtk::wmma::mma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	mtk::wmma::mma::store_matrix_sync(d, frag_d, ldd, d_layout);
}


template <int M, int N, int K, class AB_T, class C_T, class D_T, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
double get_residual(
		const typename mtk::wmma::detail::common::storage_t<AB_T>::type* const a,
		const typename mtk::wmma::detail::common::storage_t<AB_T>::type* const b,
		const C_T* const c,
		const D_T* const d
		) {
	double base_norm = 0.0;
	double diff_norm = 0.0;

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
			diff_norm += diff * diff;
			base_norm += c_v * c_v;
		}
	}
	return std::sqrt(diff_norm / base_norm);
}

std::string get_layout_name(const nvcuda::wmma::layout_t layout) {
	if (layout == nvcuda::wmma::mem_col_major) {
		return mtk::test_utils::get_string<nvcuda::wmma::col_major>();
	} else {
		return mtk::test_utils::get_string<nvcuda::wmma::row_major>();
	}
}


template <int M, int N, int K, class AB_T, class C_T, class D_T, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
void test() {
	using AB_STORAGE_T = typename mtk::wmma::detail::common::storage_t<AB_T>::type;
	D_T* d_ptr;
	C_T* c_ptr;
	AB_STORAGE_T* a_ptr;
	AB_STORAGE_T* b_ptr;

	cudaMallocHost(&a_ptr, M * K * sizeof(AB_STORAGE_T));
	cudaMallocHost(&b_ptr, K * N * sizeof(AB_STORAGE_T));
	cudaMallocHost(&c_ptr, M * N * sizeof(C_T));
	cudaMallocHost(&d_ptr, M * N * sizeof(D_T));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (std::size_t i = 0; i < M * K; i++) {
		a_ptr[i] = mtk::wmma::detail::common::cast<AB_STORAGE_T>(dist(mt));
	}
	for (std::size_t i = 0; i < K * N; i++) {
		b_ptr[i] = mtk::wmma::detail::common::cast<AB_STORAGE_T>(dist(mt));
	}
	for (std::size_t i = 0; i < M * N; i++) {
		c_ptr[i] = mtk::wmma::detail::common::cast<C_T>(dist(mt));
	}

	cudaDeviceSynchronize();
	test_kernel<M, N, K, AB_T, C_T, D_T, a_layout, b_layout, c_layout, d_layout><<<1, 32>>>(d_ptr, a_ptr, b_ptr, c_ptr);
	cudaDeviceSynchronize();
	std::printf("[TEST] M=%2d, N=%2d, K=%2d, a_%5s_%s, b_%5s_%s, c_%5s_%s, d_%5s_%s : res = %e\n",
			M, N, K,
			mtk::test_utils::get_string<AB_T>().c_str(), mtk::test_utils::get_string<a_layout>().c_str(),
			mtk::test_utils::get_string<AB_T>().c_str(), mtk::test_utils::get_string<b_layout>().c_str(),
			mtk::test_utils::get_string<C_T >().c_str(), get_layout_name(c_layout).c_str(),
			mtk::test_utils::get_string<D_T >().c_str(), get_layout_name(d_layout).c_str(),
			get_residual<M, N, K, AB_T, C_T, D_T, a_layout, b_layout, c_layout, d_layout>(a_ptr, b_ptr, c_ptr, d_ptr)
			);
}

int main() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch   : %d\n", TEST_ARCH);

#if TEST_ARCH == 80 || TEST_ARCH == 86
	test<16, 8, 16, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<16, 8, 16, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<16, 8, 16, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<16, 8, 16, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
#endif

#if TEST_ARCH == 80 || TEST_ARCH == 86 || TEST_ARCH == 75
	test<16, 8, 8, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<16, 8, 8, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<16, 8, 8, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<16, 8, 8, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
#endif

#if TEST_ARCH == 70 || TEST_ARCH == 75
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, float, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>();
	test<8, 8, 4, half, half , half , nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();
#endif
}

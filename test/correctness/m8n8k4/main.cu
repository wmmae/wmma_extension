#include <iostream>
#include <type_traits>
#include <random>
#include <mma.h>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr unsigned M = 8;
constexpr unsigned N = 8;
constexpr unsigned K = 4;

template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
__global__ void m8n8k4_test_kernel(T* const d, const half* const a, const half* const b, const S* const c) {
	mtk::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, a_layout> frag_a;
	mtk::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, b_layout> frag_b;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T> frag_c;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, S> frag_d;

	mtk::wmma::load_matrix_sync(frag_a, a, M);
	mtk::wmma::load_matrix_sync(frag_b, b, K);
	mtk::wmma::load_matrix_sync(frag_c, c, M, c_layout);

	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	mtk::wmma::store_matrix_sync(d, frag_d, M, d_layout);
}


template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
double get_residual(const half* const a, const half* const b, const S* const c, const T* const d) {
	double base_norm = 0.0;
	double diff_norm = 0.0;

	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double c_v = 0.0;
			for (unsigned k = 0; k < K; k++) {
				double a_v, b_v;
				if (std::is_same<a_layout, nvcuda::wmma::col_major>::value) {
					a_v = a[k * M + m];
				} else {
					a_v = a[k + N * m];
				}
				if (std::is_same<b_layout, nvcuda::wmma::col_major>::value) {
					b_v = b[k + K * n];
				} else {
					b_v = b[k * N + n];
				}
				c_v += a_v * b_v;
			}
			if (c_layout == nvcuda::wmma::mem_col_major) {
				c_v += c[m + M * n];
			} else {
				c_v += c[m * N + n];
			}

			// compute error
			double d_v;
			if (d_layout == nvcuda::wmma::mem_col_major) {
				d_v = d[m + M * n];
			} else {
				d_v = d[m * N + n];
			}
			const auto diff = d_v - c_v;

			// accumulate
			diff_norm += diff * diff;
			base_norm += c_v * c_v;
		}
	}
	return std::sqrt(diff_norm / base_norm);
}

template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
void test() {
	T* d_ptr;
	S* c_ptr;
	half* a_ptr;
	half* b_ptr;

	cudaMallocHost(&a_ptr, N * N * sizeof(half));
	cudaMallocHost(&b_ptr, N * N * sizeof(half));
	cudaMallocHost(&c_ptr, N * N * sizeof(T));
	cudaMallocHost(&d_ptr, N * N * sizeof(S));

	for (std::size_t i = 0; i < M * K; i++) {
		a_ptr[i] = __float2half(static_cast<float>(i + 1) / (M * K));
	}
	for (std::size_t i = 0; i < K * N; i++) {
		b_ptr[i] = __float2half(static_cast<float>(i + 1) / (K * N));
	}

	cudaDeviceSynchronize();
	m8n8k4_test_kernel<T, S, a_layout, b_layout, c_layout, d_layout><<<1, 32>>>(d_ptr, a_ptr, b_ptr, c_ptr);
	cudaDeviceSynchronize();
}

#define TEST(c_t, d_t) \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>();

int main() {
	std::printf("-- direct_product test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);

	TEST(float, float);
	TEST(half , float);
	TEST(half , half );
}

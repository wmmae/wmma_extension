#include <iostream>
#include <mma.h>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr unsigned M = 8;
constexpr unsigned N = 8;
constexpr unsigned K = 4;

template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout>
__global__ void m8n8k4_test_kernel(T* const d, const half* const a, const half* const b, const S* const c) {
	mtk::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, a_layout> frag_a;
	mtk::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, b_layout> frag_b;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T> frag_c;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, S> frag_d;

	mtk::wmma::load_matrix_sync(frag_a, a, M);
	mtk::wmma::load_matrix_sync(frag_b, b, K);
	mtk::wmma::fill_fragment(frag_c, static_cast<T>(0.0f));

	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	mtk::wmma::store_matrix_sync(d, frag_d, M, c_layout);
}

template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout>
void test() {
	T* c_ptr;
	S* d_ptr;
	half* a_ptr;
	half* b_ptr;

	cudaMallocHost(&a_ptr, N * N * sizeof(half));
	cudaMallocHost(&b_ptr, N * N * sizeof(half));
	cudaMallocHost(&c_ptr, N * N * sizeof(float));
	cudaMallocHost(&d_ptr, N * N * sizeof(float));

	for (std::size_t i = 0; i < M * K; i++) {
		a_ptr[i] = __float2half(static_cast<float>(i + 1) / (M * K));
	}
	for (std::size_t i = 0; i < K * N; i++) {
		b_ptr[i] = __float2half(static_cast<float>(i + 1) / (K * N));
	}

	cudaDeviceSynchronize();
	m8n8k4_test_kernel<T, S, a_layout, b_layout, c_layout><<<1, 32>>>(c_ptr, a_ptr, b_ptr, d_ptr);
	cudaDeviceSynchronize();
}

#define TEST(c_t, d_t) \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major>();

int main() {
	std::printf("-- direct_product test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);

	TEST(float, float);
	TEST(half , float);
	TEST(half , half );
}

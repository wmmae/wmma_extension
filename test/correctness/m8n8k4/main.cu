#include <iostream>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr std::size_t M = 8;
constexpr std::size_t N = 8;
constexpr std::size_t K = 4;


template <class T, class S, class a_layout, class b_layout>
__global__ void m8n8k4_test_kernel(T* const d, const half* const a, const half* const b, const S* const c) {
	mtk::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, a_layout> frag_a;
	mtk::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, b_layout> frag_b;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T> frag_c;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, S> frag_d;

	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_d);
}

template <class T, class S, class a_layout, class b_layout>
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
	m8n8k4_test_kernel<T, S, a_layout, b_layout><<<1, 32>>>(c_ptr, a_ptr, b_ptr, d_ptr);
	cudaDeviceSynchronize();
}

int main() {
	std::printf("-- direct_product test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);

	test<float, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major>();
}

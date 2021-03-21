#include <iostream>
#include <wmma_extension/wmma_mma.hpp>
#include "common.hpp"

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

__device__ float to_float(const float a) {return a;}
__device__ float to_float(const half  a) {return __half2float(a);}

template <class Use, int M, int N, int K, class T, class Layout>
__global__ void fill_test_kernel() {
	constexpr float a = 2.f;
	mtk::wmma::mma::fragment<Use, M, N, K, T, Layout> frag_zero, frag_a;
	mtk::wmma::mma::fill_zero(frag_zero);
	mtk::wmma::mma::fill_fragment(frag_a, a);

	float max_error_z = 0.f;
	for (unsigned i = 0; i < frag_zero.num_elements; i++) {
		max_error_z = max(abs(to_float(frag_zero.x[i])), max_error_z);
	}

	float max_error_a = 0.f;
	for (unsigned i = 0; i < frag_a.num_elements; i++) {
		max_error_a = max(abs(to_float(frag_a.x[i]) - a), max_error_a);
	}

	printf("[%3u] E(a)=%e [%6s], E(z)=%e [%6s]\n",
			threadIdx.x,
			max_error_a,
			(max_error_a < 1e-6 ? "PASSED" : "FAILED"),
			max_error_z,
			(max_error_z < 1e-6 ? "PASSED" : "FAILED")
			);
}

template <class Use, int M, int N, int K, class T, class Layout>
void test() {
	std::printf("[TEST] %11s, %d, %d, %d, %5s, %8s\n",
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<T>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str()
			);
	fill_test_kernel<Use, M, N, K, T, Layout><<<1, 32>>>();
	cudaDeviceSynchronize();
}

int main() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch    : %d\n", TEST_ARCH);

#if TEST_ARCH == 80 || TEST_ARCH == 86
	test<nvcuda::wmma::matrix_a   , 16, 8, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 16, 8, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 16, half, void                   >();
#endif
#if TEST_ARCH == 80 || TEST_ARCH == 86 || TEST_ARCH == 75
	test<nvcuda::wmma::matrix_a   , 16, 8, 8 , half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 16, 8, 8 , half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 8 , half, void                   >();
#endif
#if TEST_ARCH == 75 || TEST_ARCH ==70 
	test<nvcuda::wmma::matrix_a   , 8, 8, 4 , half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a   , 8, 8, 4 , half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 8, 8, 4 , half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b   , 8, 8, 4 , half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 8, 8, 4 , half, void                   >();
#endif
}

#include <iostream>
#include <wmma_extension/wmma_mma.hpp>
#include "common.hpp"

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

__device__ float to_float(const float a) {return a;}
__device__ float to_float(const half  a) {return __half2float(a);}

__device__ float my_fabs(const float a) {
	return a > 0.f ? a : -a;
}

template <class Use, int M, int N, int K, class T, class Layout>
__global__ void fill_test_kernel(float* const g_max_error_a, float* const g_max_error_z) {
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

	if (threadIdx.x == 0) {
		*g_max_error_a = 0;
		*g_max_error_z = 0;
	}
	__syncthreads();
	for (unsigned i = 0; i < blockDim.x; i++) {
		if (threadIdx.x == i) {
			*g_max_error_a = max(*g_max_error_a, my_fabs(max_error_a));
			*g_max_error_z = max(*g_max_error_z, my_fabs(max_error_z));
		}
		__syncthreads();
	}
}

template <class Use, int M, int N, int K, class T, class Layout>
void test() {
	float *max_error_a;
	float *max_error_z;
	cudaMallocHost(&max_error_a, sizeof(float));
	cudaMallocHost(&max_error_z, sizeof(float));
	fill_test_kernel<Use, M, N, K, T, Layout><<<1, 32>>>(max_error_a, max_error_z);
	cudaDeviceSynchronize();
	std::printf("[%s] ARCH=%d, %11s, %2d, %2d, %2d, %5s, %10s, fill_zero_error=%e [%s], fill_a_error=%e, [%s]\n",
			__FILE__,
			TEST_ARCH,
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<T>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str(),
			*max_error_a,
			mtk::test_utils::get_test_result_string((*max_error_a) < mtk::test_utils::get_machine_eps<T>()),
			*max_error_z,
			mtk::test_utils::get_test_result_string((*max_error_z) < mtk::test_utils::get_machine_eps<T>())
			);
	cudaFreeHost(max_error_a);
	cudaFreeHost(max_error_z);
}

int main() {
#if TEST_ARCH >= 80
	test<nvcuda::wmma::matrix_a   , 16, 8, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 16, 8, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 16, half, void                   >();
#endif
#if TEST_ARCH >= 75
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

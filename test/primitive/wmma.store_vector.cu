#include <iostream>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>
#include "common.hpp"

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

//#define TEST_TF32

#ifndef TEST_TF32
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
#else
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 8;
#endif

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

__global__ void test_store_vector_kernel(
		float* const dst,
		const float* const src,
		const nvcuda::wmma::layout_t layout
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> frag_c;
	nvcuda::wmma::load_matrix_sync(frag_c, src, M, layout);
	mtk::wmma::store_vector(dst, frag_c, layout);
}

void test(const nvcuda::wmma::layout_t layout) {
	float* src_mem;
	float* dst_mem;

	cudaMallocHost(&src_mem, M * N * sizeof(float));
	cudaMallocHost(&dst_mem, M * sizeof(float));

	for (std::size_t i = 0; i < M * N; i++) {
		src_mem[i] = static_cast<float>(i);
	}

	cudaDeviceSynchronize();
	test_store_vector_kernel<<<1, 32>>>(dst_mem, src_mem, layout);
	cudaDeviceSynchronize();

	double error = 0;
	for (std::size_t i = 0; i < M; i++) {
		const double diff = src_mem[i] - dst_mem[i];
		error = std::max(error, std::abs(diff));
	}

	cudaFreeHost(src_mem);
	cudaFreeHost(dst_mem);

	std::printf("[%s] ARCH=%d, <%2d, %2d, %2d>, error=%e, [%s]\n",
			__FILE__,
			TEST_ARCH,
			M, N, K,
			error,
			mtk::test_utils::get_test_result_string(error < mtk::test_utils::get_machine_eps<float>())
			);
}

int main() {
	test(nvcuda::wmma::mem_row_major);
	test(nvcuda::wmma::mem_col_major);
	test(nvcuda::wmma::mem_row_major);
	test(nvcuda::wmma::mem_col_major);
}

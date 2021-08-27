#include <iostream>
#include <cmath>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>
#include "common.hpp"

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

__global__ void test_gevm_kernel(
		float* const dst,
		const half* const src,
		const half* const eye
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> eye_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> vec_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> result_frag;
	nvcuda::wmma::load_matrix_sync(eye_frag, eye, 16);
	nvcuda::wmma::fill_fragment(result_frag, 0.0f);

	mtk::wmma::load_vector(vec_frag, src);

	nvcuda::wmma::mma_sync(result_frag, eye_frag, vec_frag, result_frag);

	mtk::wmma::store_vector(dst, result_frag, nvcuda::wmma::mem_col_major);
}

void test() {
	half* src_mem;
	float* dst_mem;
	half* eye_mem;

	cudaMallocHost(&src_mem, 16 * 16 * sizeof(half));
	cudaMallocHost(&dst_mem, 16 * sizeof(float));
	cudaMallocHost(&eye_mem, 16 * 16 * sizeof(half));

	for (std::size_t i = 0; i < 16 * 16; i++) {
		src_mem[i] = convert<half, float>((i < 16) ? i : 0);
		eye_mem[i] = convert<half>((i % 17 == 0) ? 1.0f : 0.0f);
	}

	cudaDeviceSynchronize();
	test_gevm_kernel<<<1, 32>>>(dst_mem, src_mem, eye_mem);
	cudaDeviceSynchronize();

	double error = 0.;
	for (std::size_t i = 0; i < 16; i++) {
		const double diff = convert<float, half>(dst_mem[i]) - convert<float, half>(src_mem[i]);
		error = std::max(error, std::abs(diff));
	}
	std::printf("[%s] ARCH=%d, error=%e [%s]\n",
			__FILE__,
			TEST_ARCH,
			error,
			mtk::test_utils::get_test_result_string(error < mtk::test_utils::get_machine_eps<float>() * 16)
			);
}

int main() {
	test();
}

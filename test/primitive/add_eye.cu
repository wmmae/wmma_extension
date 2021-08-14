#include <iostream>
#include <random>
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

template <class T, int M, int N, int K>
__global__ void make_eye_kernel(T* const eye, const T a) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, M, K, T> frag_c;
	nvcuda::wmma::fill_fragment(frag_c, convert<T>(1.0f));
	mtk::wmma::add_eye(frag_c, a);
	nvcuda::wmma::store_matrix_sync(eye, frag_c, N, nvcuda::wmma::mem_col_major);
}

template <class T, int M, int N, int K>
void test() {
	T *h;

	cudaMallocHost(&h, sizeof(T) * N * N);

	cudaDeviceSynchronize();
	make_eye_kernel<T, M, N, K><<<1, 32>>>(h, convert<T>(2.0f));
	cudaDeviceSynchronize();

	double max_error = 0.0;
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			const float c = (i == j) ? 3.0f : 1.0f;
			const double diff = c - convert<float>(h[i * 16 + j]);
			max_error = std::max(max_error, std::abs(diff));
		}
	}
	std::printf("[%s] arch=%d, error=%e, [%s]\n",
			__FILE__,
			TEST_ARCH,
			max_error,
			mtk::test_utils::get_test_result_string(max_error < mtk::test_utils::get_machine_eps<T>())
			);
}

int main() {
	test<float, 16, 16, 16>();
	test<half , 16, 16, 16>();
#ifdef TEST_TF32
	test<float, 16, 16, 8 >();
#endif
}

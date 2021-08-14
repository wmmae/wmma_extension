#include <iostream>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <class layout>
__global__ void test_load_vector_kernel(
		const half* const src
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, layout> frag;
	nvcuda::wmma::load_matrix_sync(frag, src, 16);

	mtk::wmma::print_fragment(frag);
}

template <class layout>
void test() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch   : %d\n", TEST_ARCH);
	if (std::is_same<layout, nvcuda::wmma::col_major>::value) {
		std::printf("layout : col_major\n");
	} else {
		std::printf("layout : row_major\n");
	}
	half* src_mem;

	cudaMallocHost(&src_mem, 16 * sizeof(half));

	for (std::size_t i = 0; i < 16 * 16; i++) {
		src_mem[i] = convert<half, float>(i);
	}

	cudaDeviceSynchronize();
	test_load_vector_kernel<layout><<<1, 32>>>(src_mem);
	cudaDeviceSynchronize();
}

int main() {
	test<nvcuda::wmma::row_major>();
	test<nvcuda::wmma::col_major>();
}

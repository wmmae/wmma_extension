#include <iostream>
#include <type_traits>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <class T>
__global__ void test_store_vector_kernel(
		T* const dst,
		const T* const src,
		const nvcuda::wmma::layout_t layout
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T> frag_c;
	nvcuda::wmma::load_matrix_sync(frag_c, src, 16, layout);
	mtk::wmma::store_vector_sync(dst, frag_c, layout);
}

template <class T>
void test(const nvcuda::wmma::layout_t layout) {
	std::printf("-- store_vector test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);
	if (layout == nvcuda::wmma::mem_col_major) {
		std::printf("layout : col_major\n");
	} else {
		std::printf("layout : row_major\n");
	}
	if (std::is_same<float, T>::value)
		std::printf("type   : float\n");
	if (std::is_same<half, T>::value)
		std::printf("type   : half\n");
	T* src_mem;
	T* dst_mem;

	cudaMallocHost(&src_mem, 16 * 16 * sizeof(T));
	cudaMallocHost(&dst_mem, 16 * sizeof(T));

	for (std::size_t i = 0; i < 16 * 16; i++) {
			src_mem[i] = convert<T, float>(i);
	}

	cudaDeviceSynchronize();
	test_store_vector_kernel<<<1, 32>>>(dst_mem, src_mem, layout);
	cudaDeviceSynchronize();

	for (std::size_t i = 0; i < 16; i++) {
		std::printf("%3.1f ", convert<float, T>(dst_mem[i]));
	}
	std::printf("\n");
}

int main() {
	test<float>(nvcuda::wmma::mem_row_major);
	test<float>(nvcuda::wmma::mem_col_major);
	test<half >(nvcuda::wmma::mem_row_major);
	test<half >(nvcuda::wmma::mem_col_major);
}

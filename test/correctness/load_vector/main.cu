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

template <class T, class layout>
__global__ void test_load_vector_kernel(
		float* const dst,
		const T* const src,
		const half* const eye
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> eye_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, layout> vec_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> result_frag;
	nvcuda::wmma::load_matrix_sync(eye_frag, eye, 16);
	nvcuda::wmma::fill_fragment(result_frag, 0.0f);

	mtk::wmma::load_vector_sync(vec_frag, src);

	nvcuda::wmma::mma_sync(result_frag, eye_frag, vec_frag, result_frag);
	nvcuda::wmma::store_matrix_sync(dst, result_frag, 16, nvcuda::wmma::mem_col_major);
}

template <class T, class layout>
void test() {
	std::printf("-- load_vector test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);
	if (std::is_same<layout, nvcuda::wmma::col_major>::value) {
		std::printf("layout : col_major\n");
	} else {
		std::printf("layout : row_major\n");
	}
	if (std::is_same<float, T>::value)
		std::printf("type   : float\n");
	if (std::is_same<half, T>::value)
		std::printf("type   : half\n");
	T* src_mem;
	float* dst_mem;
	half* eye_mem;

	cudaMallocHost(&src_mem, 16 * sizeof(T));
	cudaMallocHost(&dst_mem, 16 * 16 * sizeof(float));
	cudaMallocHost(&eye_mem, 16 * 16 * sizeof(half));

	for (std::size_t i = 0; i < 16 * 16; i++) {
		src_mem[i] = convert<T, float>(i);
		eye_mem[i] = convert<half>((i % 17 == 0) ? 1.0f : 0.0f);
	}

	cudaDeviceSynchronize();
	test_load_vector_kernel<T, layout><<<1, 32>>>(dst_mem, src_mem, eye_mem);
	cudaDeviceSynchronize();

	for (std::size_t i = 0; i < 16; i++) {
		for (std::size_t j = 0; j < 16; j++) {
			std::printf("%03d ", static_cast<int>(convert<float, T>(dst_mem[i + j * 16])));
		}
		std::printf("\n");
	}
	std::printf("\n");
}

int main() {
	test<float, nvcuda::wmma::row_major>();
	test<float, nvcuda::wmma::col_major>();
	test<half , nvcuda::wmma::row_major>();
	test<half , nvcuda::wmma::col_major>();
}

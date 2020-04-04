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

__global__ void test_gevm_kernel(
		float* const dst,
		const half* const src,
		const half* const eye
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> vec_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> eye_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> result_frag;
	nvcuda::wmma::load_matrix_sync(eye_frag, eye, 16);
	nvcuda::wmma::fill_fragment(result_frag, 0.0f);

	mtk::wmma::load_vector_sync(vec_frag, src);

	nvcuda::wmma::mma_sync(result_frag, vec_frag, eye_frag, result_frag);

	mtk::wmma::store_vector_sync(dst, result_frag, nvcuda::wmma::mem_row_major);
}

void test() {
	std::printf("-- gevm test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);
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

	for (std::size_t i = 0; i < 16; i++) {
		std::printf("%03d ", static_cast<int>(convert<float, half>(dst_mem[i])));
	}
	std::printf("\n");
}

int main() {
	test();
}

#include <iostream>
#include <type_traits>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

//#define TEST_TF32

#ifndef TEST_TF32
constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr std::size_t K = 16;
using ab_type = half;
#else
constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr std::size_t K = 8;
using ab_type = nvcuda::wmma::precision::tf32;
#endif

using storage_t = typename mtk::wmma::detail::common::storage_t<ab_type>::type;

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <class Use, class layout>
__global__ void test_load_vector_kernel(
		const storage_t* const src
		) {
	nvcuda::wmma::fragment<Use, M, N, K, ab_type, layout> vec_frag;
	mtk::wmma::load_vector_sync(vec_frag, src);

	mtk::wmma::print_fragment(vec_frag, "vec");
}

template <class Use, class layout>
void test() {
	std::printf("-- load_vector test --\n");
	std::printf("arch   : %d\n", TEST_ARCH);
	if (std::is_same<layout, nvcuda::wmma::col_major>::value) {
		std::printf("layout : col_major\n");
	} else {
		std::printf("layout : row_major\n");
	}
	if (std::is_same<float, ab_type>::value)
		std::printf("type   : float\n");
	if (std::is_same<half, ab_type>::value)
		std::printf("type   : half\n");
	if (std::is_same<nvcuda::wmma::precision::tf32, ab_type>::value)
		std::printf("type   : tf32\n");

	if (std::is_same<nvcuda::wmma::matrix_a, Use>::value)
		std::printf("use    : a\n");
	if (std::is_same<nvcuda::wmma::matrix_b, Use>::value)
		std::printf("use    : b\n");
	std::printf("size   : %lu, %lu, %lu\n", M, N, K);

	storage_t* src_mem;

	cudaMallocHost(&src_mem, 16 * sizeof(storage_t));

	for (std::size_t i = 0; i < 16; i++) {
		src_mem[i] = convert<storage_t, float>(i);
	}

	cudaDeviceSynchronize();
	test_load_vector_kernel<Use, layout><<<1, 32>>>(src_mem);
	cudaDeviceSynchronize();
}

int main() {
	test<nvcuda::wmma::matrix_a, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a, nvcuda::wmma::row_major>();

	test<nvcuda::wmma::matrix_b, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, nvcuda::wmma::row_major>();
}

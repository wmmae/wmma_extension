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

template <class T>
__device__ T m_abs(const T a) {
	if (a >= convert<T, float>(0)) return a;
	return -1;
}

template <class Use, class layout>
__global__ void test_load_vector_kernel(
		const storage_t* const src,
		const storage_t* const cor
		) {
	nvcuda::wmma::fragment<Use, M, N, K, ab_type, layout> vec_frag;
	mtk::wmma::load_vector_sync(vec_frag, src);

	nvcuda::wmma::fragment<Use, M, N, K, ab_type, layout> cor_frag;
	mtk::wmma::load_vector_sync(cor_frag, cor);

	storage_t error = convert<storage_t, float>(0.0f);
	for (unsigned i = 0; i < vec_frag.num_elements; i++) {
		error += m_abs(vec_frag.x[i] - cor_frag.x[i]);
	}
	printf("[%2u] error = %e\n", threadIdx.x, convert<float>(error));
}

template <class Use, class layout>
void test() {
	std::size_t cor_size = 0;
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

	if (std::is_same<nvcuda::wmma::matrix_a, Use>::value) {
		std::printf("use    : a\n");
		cor_size = M * K;
	}
	if (std::is_same<nvcuda::wmma::matrix_b, Use>::value) {
		std::printf("use    : b\n");
		cor_size = N * K;
	}
	std::printf("size   : %lu, %lu, %lu\n", M, N, K);

	storage_t* src_mem;
	storage_t* cor_mem;

	cudaMallocHost(&src_mem, M * sizeof(storage_t));
	cudaMallocHost(&cor_mem, cor_size * sizeof(storage_t));

	for (std::size_t i = 0; i < cor_size; i++) {
		cor_mem[i] = convert<storage_t, float>(0);
	}

	for (std::size_t i = 0; i < 16; i++) {
		src_mem[i] = convert<storage_t, float>(i);
		cor_mem[i] = convert<storage_t, float>(i);
	}

	cudaDeviceSynchronize();
	test_load_vector_kernel<Use, layout><<<1, 32>>>(src_mem, cor_mem);
	cudaDeviceSynchronize();
}

int main() {
	test<nvcuda::wmma::matrix_a, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a, nvcuda::wmma::row_major>();

	test<nvcuda::wmma::matrix_b, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, nvcuda::wmma::row_major>();
}

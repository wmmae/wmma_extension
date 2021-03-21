#include <iostream>
#include <wmma_extension/wmma_mma.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

template <class T>
std::string get_string();
template <> std::string get_string<float>() {return "float";}
template <> std::string get_string<half >() {return "half";}
template <> std::string get_string<nvcuda::wmma::col_major>() {return "col_major";}
template <> std::string get_string<nvcuda::wmma::row_major>() {return "row_major";}
template <> std::string get_string<void>() {return "row_major";}
template <> std::string get_string<nvcuda::wmma::matrix_a>()  {return "matrix_a";}
template <> std::string get_string<nvcuda::wmma::matrix_b>()  {return "matrix_b";}
template <> std::string get_string<nvcuda::wmma::accumulator>()  {return "accumulator";}

__device__ float to_float(const float a) {return a;}
__device__ float to_float(const half  a) {return __half2float(a);}

template <class Use, int M, int N, int K, class T, class Layout>
__global__ void fill_test_kernel() {
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

	printf("[%3u] E(a)=%e [%6s], E(z)=%e [%6s]\n",
			threadIdx.x,
			max_error_a,
			(max_error_a < 1e-6 ? "PASSED" : "FAILED"),
			max_error_z,
			(max_error_z < 1e-6 ? "PASSED" : "FAILED")
			);
}

template <class Use, int M, int N, int K, class T, class Layout>
void test() {
	std::printf("[TEST] %11s, %d, %d, %d, %5s, %8s\n",
			get_string<Use>().c_str(),
			M, N, K,
			get_string<T>().c_str(),
			get_string<Layout>().c_str()
			);
	fill_test_kernel<Use, M, N, K, T, Layout><<<1, 32>>>();
	cudaDeviceSynchronize();
}

int main() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch    : %d\n", TEST_ARCH);

#if TEST_ARCH == 80 || TEST_ARCH == 86
	test<nvcuda::wmma::matrix_a   , 16, 8, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 16, 8, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 16, half, void                   >();
#endif
#if TEST_ARCH == 80 || TEST_ARCH == 86 || TEST_ARCH == 75
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

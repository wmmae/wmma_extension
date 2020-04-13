#include <iostream>
#include <random>
#include <type_traits>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr std::size_t N = 16;

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <unsigned CORRECTION_TERMS>
__global__ void direct_product_kernel(float* const h, const float* const u) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::row_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, N, N, float> frag_c;

	__shared__ half su[N];
	__shared__ half sdu[N];

	if (threadIdx.x < N) {
		const auto fv = u[threadIdx.x];
		const auto hv = convert<half>(fv);
		su[threadIdx.x] = hv;
		sdu[threadIdx.x] = convert<half>(fv - convert<float>(hv));
	}

	if (CORRECTION_TERMS == 3) {
		mtk::wmma::make_direct_product_fragment_c3(
				frag_a,
				su, sdu
				);
		mtk::wmma::make_direct_product_fragment_c3(
				frag_b,
				su, sdu
				);
	} else {
		mtk::wmma::make_direct_product_fragment(
				frag_a,
				su, sdu
				);
		mtk::wmma::make_direct_product_fragment(
				frag_b,
				su, sdu
				);
	}

	nvcuda::wmma::fill_fragment(frag_c, 0.0f);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(h, frag_c, N, nvcuda::wmma::mem_col_major);
}

template <unsigned CORRECTION_TERMS>
void test() {
	std::printf("-- direct_product test --\n");
	std::printf("arch    : %d\n", TEST_ARCH);
	std::printf("c terms : %u\n", CORRECTION_TERMS);
	float *u;
	float *h;

	cudaMallocHost(&u, sizeof(float) * N);
	cudaMallocHost(&h, sizeof(float) * N * N);

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (unsigned i = 0; i < N; i++) {
		u[i] = dist(mt);
	}

	cudaDeviceSynchronize();
	direct_product_kernel<CORRECTION_TERMS><<<1, 32>>>(h, u);
	cudaDeviceSynchronize();

	double max_error = 0.0;
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			const double diff = static_cast<double>(u[i]) * static_cast<double>(u[j]) - static_cast<double>(h[i * N + j]);
			max_error = std::max(max_error, std::abs(diff));
		}
	}
	std::printf("error   : %e\n", max_error);
}

int main() {
	test<2>();
	test<3>();
}

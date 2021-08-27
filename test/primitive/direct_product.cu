#include <iostream>
#include <random>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>
#include "common.hpp"

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

// #define SMALLER_WORKING_MEMORY

using storage_t = typename mtk::wmma::detail::common::storage_t<ab_type>::type;

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <unsigned CORRECTION_TERMS>
__global__ void direct_product_kernel(float* const h, const float* const u, const float* const v) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a   , M, N, K, ab_type, nvcuda::wmma::col_major> frag_a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b   , M, N, K, ab_type, nvcuda::wmma::row_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> frag_c;

	__shared__ storage_t su[N];
	__shared__ storage_t sdu[N];
	__shared__ storage_t sv[N];
	__shared__ storage_t sdv[N];

	if (threadIdx.x < N) {
		const auto fv = u[threadIdx.x];
		const auto hv = mtk::wmma::detail::common::cast<ab_type>(fv);
		su[threadIdx.x] = hv;
		sdu[threadIdx.x] = convert<storage_t>(fv - convert<float>(hv));
	} else {
		const auto fv = v[threadIdx.x - N];
		const auto hv = mtk::wmma::detail::common::cast<ab_type>(fv);
		sv[threadIdx.x - N] = hv;
		sdv[threadIdx.x - N] = convert<storage_t>(fv - convert<float>(hv));
	}

	__syncthreads();

	if (CORRECTION_TERMS == 3) {
#ifdef SMALLER_WORKING_MEMORY
		mtk::wmma::make_direct_product_fragment_c3(
				frag_a,
				u
				);
		mtk::wmma::make_direct_product_fragment_c3(
				frag_b,
				v
				);
#else
		mtk::wmma::make_direct_product_fragment_c3(
				frag_a,
				su, sdu
				);
		mtk::wmma::make_direct_product_fragment_c3(
				frag_b,
				sv, sdv
				);
#endif
	} else {
#ifdef SMALLER_WORKING_MEMORY
		mtk::wmma::make_direct_product_fragment(
				frag_a,
				u
				);
		mtk::wmma::make_direct_product_fragment(
				frag_b,
				v
				);
#else
		mtk::wmma::make_direct_product_fragment(
				frag_a,
				su, sdu
				);
		mtk::wmma::make_direct_product_fragment(
				frag_b,
				sv, sdv
				);
#endif
	}

	nvcuda::wmma::fill_fragment(frag_c, 0.0f);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(h, frag_c, N, nvcuda::wmma::mem_col_major);
}

template <unsigned CORRECTION_TERMS>
void test() {
	float *u;
	float *v;
	float *h;

	cudaMallocHost(&u, sizeof(float) * N);
	cudaMallocHost(&v, sizeof(float) * N);
	cudaMallocHost(&h, sizeof(float) * N * N);

	std::mt19937 mt(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (unsigned i = 0; i < N; i++) {
		u[i] = dist(mt);
		v[i] = dist(mt);
	}

	cudaDeviceSynchronize();
	direct_product_kernel<CORRECTION_TERMS><<<1, 32>>>(h, u, v);
	cudaDeviceSynchronize();

	double max_error = 0.0;
	for (unsigned i = 0; i < M; i++) {
		for (unsigned j = 0; j < N; j++) {
			const double diff = static_cast<double>(u[i]) * static_cast<double>(v[j]) - static_cast<double>(h[i + N * j]);
			max_error = std::max(max_error, std::abs(diff));
		}
	}
	std::printf("[%s] ARCH=%d, c_terms=%u, error=%e [%s]\n",
			__FILE__,
			TEST_ARCH,
			CORRECTION_TERMS,
			max_error,
			mtk::test_utils::get_test_result_string(max_error < mtk::test_utils::get_machine_eps<float>() * 16)
			);

	cudaFreeHost(u);
	cudaFreeHost(v);
}

int main() {
	test<2>();
	test<3>();
}

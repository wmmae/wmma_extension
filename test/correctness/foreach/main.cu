#include <iostream>
#include <random>
#include <type_traits>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr std::size_t K = 16;

using ab_type = half;

__global__ void matmul16x16_kernel(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, ab_type, nvcuda::wmma::col_major> frag_a, frag_da;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, ab_type, nvcuda::wmma::col_major> frag_b, frag_db;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> frag_c;

	mtk::wmma::fill_zero(frag_c);

	mtk::wmma::foreach(frag_a,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto a = a_ptr[mem_index];
				const auto a_rp = mtk::wmma::detail::common::cast<ab_type>(a);
				frag_a.x[frag_index] = a_rp;
				frag_da.x[frag_index] = mtk::wmma::detail::common::cast<ab_type>(a - mtk::wmma::detail::common::cast<float>(a_rp));
			});

	mtk::wmma::foreach(frag_b,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto b = b_ptr[mem_index];
				const auto b_rp = mtk::wmma::detail::common::cast<ab_type>(b);
				frag_b.x[frag_index] = b_rp;
				frag_db.x[frag_index] = mtk::wmma::detail::common::cast<ab_type>(b - mtk::wmma::detail::common::cast<float>(b_rp));
			});

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_db, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_da, frag_b, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(c_ptr, frag_c, N, nvcuda::wmma::mem_col_major);
}

void test() {
	std::printf("-- foreach test --\n");
	std::printf("arch    : %d\n", TEST_ARCH);

	float *a, *b, *c;
	cudaMallocHost(&a, sizeof(float) * N * N);
	cudaMallocHost(&b, sizeof(float) * N * N);
	cudaMallocHost(&c, sizeof(float) * N * N);

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (unsigned i = 0; i < N * N; i++) {
		a[i] = dist(mt);
		b[i] = dist(mt);
		c[i] = static_cast<float>(0);
	}

	cudaDeviceSynchronize();
	matmul16x16_kernel<<<1, 32>>>(c, a, b);
	cudaDeviceSynchronize();

	double max_error = 0.0;
	for (unsigned i = 0; i < M; i++) {
		for (unsigned j = 0; j < N; j++) {
			double sum = 0.0;
			for (unsigned k = 0; k < K; k++) {
				sum += static_cast<double>(a[i + M * k]) * static_cast<double>(b[k + j * K]);
			}
			const auto error = std::abs(sum - c[i + j * M]);
			std::printf("%e ", error);
			max_error = std::max(max_error, error);
		}
		std::printf("\n");
	}
	std::printf("error   : %e\n", max_error);
}

int main() {
	test();
}

#include <iostream>
#include <random>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

// #define TEST_TF32

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

__global__ void matmul16x16_kernel(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, ab_type, nvcuda::wmma::col_major> frag_a, frag_da;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, ab_type, nvcuda::wmma::col_major> frag_b, frag_db;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> frag_c;

	mtk::wmma::foreach<decltype(frag_c)>(
			nvcuda::wmma::mem_col_major,
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				const auto c = c_ptr[mem_index];
				for (unsigned i = 0; i < frag_index_count; i++) {
					const unsigned frag_index = frag_index_list[i];
					frag_c.x[frag_index] = c;
				}
			});

	mtk::wmma::foreach<decltype(frag_a)>(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				const auto a = a_ptr[mem_index];
				const auto a_rp = mtk::wmma::detail::common::cast<ab_type>(a);
				const auto da_rp = mtk::wmma::detail::common::cast<ab_type>(a - mtk::wmma::detail::common::cast<float>(a_rp));
				for (unsigned i = 0; i < frag_index_count; i++) {
					const unsigned frag_index = frag_index_list[i];
					frag_a.x[frag_index] = a_rp;
					frag_da.x[frag_index] = da_rp;
				}
			});

	mtk::wmma::foreach<decltype(frag_b)>(
			[&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				const auto b = b_ptr[mem_index];
				const auto b_rp = mtk::wmma::detail::common::cast<ab_type>(b);
				const auto db_rp = mtk::wmma::detail::common::cast<ab_type>(b - mtk::wmma::detail::common::cast<float>(b_rp));
				for (unsigned i = 0; i < frag_index_count; i++) {
					const unsigned frag_index = frag_index_list[i];
					frag_b.x[frag_index] = b_rp;
					frag_db.x[frag_index] = db_rp;
				}
			});

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_db, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_da, frag_b, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(c_ptr, frag_c, N, nvcuda::wmma::mem_col_major);
}

void test() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch    : %d\n", TEST_ARCH);

	float *a, *b, *c, *d;
	cudaMallocHost(&a, sizeof(float) * N * N);
	cudaMallocHost(&b, sizeof(float) * N * N);
	cudaMallocHost(&c, sizeof(float) * N * N);
	cudaMallocHost(&d, sizeof(float) * N * N);

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (unsigned i = 0; i < N * N; i++) {
		a[i] = dist(mt);
		b[i] = dist(mt);
		d[i] = c[i] = dist(mt);
	}

	cudaDeviceSynchronize();
	matmul16x16_kernel<<<1, 32>>>(c, a, b);
	cudaDeviceSynchronize();

	double max_error = 0.0;
	double max_element = 0.0;
	for (unsigned i = 0; i < M; i++) {
		for (unsigned j = 0; j < N; j++) {
			double sum = d[i + j * M];
			for (unsigned k = 0; k < K; k++) {
				sum += static_cast<double>(a[i + M * k]) * static_cast<double>(b[k + j * K]);
			}
			const auto error = std::abs(sum - c[i + j * M]);
			const auto element = std::abs(sum);
			max_error = std::max(max_error, error);
			max_element = std::max(max_element, element);
		}
	}
	const auto e = max_error / max_element;
	std::printf("{%s} error=%e [",
			__FILE__,
			e);
	if (e < 1e-6) {
		std::printf("PASSED");
	} else {
		std::printf("FAILED");
	}
	std::printf("]\n");
}

int main() {
	test();
}

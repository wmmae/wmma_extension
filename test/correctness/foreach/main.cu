#include <iostream>
#include <random>
#include <type_traits>
#include <wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr std::size_t N = 16;

__global__ void matmul16x16_kernel(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a, frag_da;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::col_major> frag_b, frag_db;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, N, N, float> frag_c;

	mtk::wmma::fill_zero(frag_c);

	mtk::wmma::foreach(frag_a,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto a = a_ptr[mem_index];
				const auto a_fp16 = __float2half(a);
				frag_a.x[frag_index] = a_fp16;
				frag_da.x[frag_index] = __float2half(a - __half2float(a_fp16));
			});

	mtk::wmma::foreach(frag_b,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto b = b_ptr[mem_index];
				const auto b_fp16 = __float2half(b);
				frag_b.x[frag_index] = b_fp16;
				frag_db.x[frag_index] = __float2half(b - __half2float(b_fp16));
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
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			double sum = 0.0;
			for (unsigned k = 0; k < N; k++) {
				sum += static_cast<double>(a[i + N * k]) * static_cast<double>(b[k + j * N]);
			}
			const auto error = std::abs(sum - c[i + j * N]);
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

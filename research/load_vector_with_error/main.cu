#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <wmma_extension.hpp>

constexpr unsigned N = 16;
constexpr unsigned NUM_SAMPLES = 1u << 15;

enum TEST_MODE {
	SINGLE_MMA,
	THREE_MMA
};

__global__ void load_vector_with_error_kernel(float* const h, const float* const u) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::row_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, N, N, float> frag_c;

	mtk::wmma::foreach(
			frag_a,
			[&frag_a, &u](const unsigned frag_index, const unsigned mem_index) {
				if (mem_index < 2 * N) {
					frag_a.x[frag_index] = __float2half(u[mem_index & 0xf]);
				} else if (mem_index < 3 * N) {
					frag_a.x[frag_index] = __float2half(u[mem_index & 0xf] - __half2float(__float2half(u[mem_index & 0xf])));
				} else {
					frag_a.x[frag_index] = __float2half(0.0f);
				}
			});
    
	mtk::wmma::foreach(
			frag_b,
			[&frag_b, &u](const unsigned frag_index, const unsigned mem_index) {
				if (mem_index < N) {
					frag_b.x[frag_index] = __float2half(u[mem_index & 0xf]);
				} else if (mem_index < 2 * N) {
					frag_b.x[frag_index] = __float2half(u[mem_index & 0xf] - __half2float(__float2half(u[mem_index & 0xf])));
				} else if (mem_index < 3 * N) {
					frag_b.x[frag_index] = __float2half(u[mem_index & 0xf]);
				} else {
					frag_b.x[frag_index] = __float2half(0.0f);
				}
			});

	nvcuda::wmma::fill_fragment(frag_c, 0.0f);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(h, frag_c, N, nvcuda::wmma::mem_col_major);
}

__global__ void load_vector_kernel(float* const h, const float* const u) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a, frag_a_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::row_major> frag_b, frag_b_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, N, N, float> frag_c;

	__shared__ float su[N];
	__syncthreads();
	if (threadIdx.x < N) su[threadIdx.x] = u[threadIdx.x];
	__syncthreads();

	mtk::wmma::load_vector_sync(frag_a, su);
	mtk::wmma::load_vector_sync(frag_b, su);

	__syncthreads();
	if (threadIdx.x < N) su[threadIdx.x] = u[threadIdx.x] - __half2float(__float2half(u[threadIdx.x]));;
	__syncthreads();

	mtk::wmma::load_vector_sync(frag_a_diff, su);
	mtk::wmma::load_vector_sync(frag_b_diff, su);


	nvcuda::wmma::fill_fragment(frag_c, 0.0f);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_a_diff, frag_b, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b_diff, frag_c);
	nvcuda::wmma::mma_sync(frag_c, frag_a_diff, frag_b_diff, frag_c);

	nvcuda::wmma::store_matrix_sync(h, frag_c, N, nvcuda::wmma::mem_col_major);
}

void test(
		const TEST_MODE test_mode
		) {
	std::printf("----------------\n");
	if (test_mode == TEST_MODE::THREE_MMA) {
		std::printf("%20s : %s\n", "TEST_MODE", "SINGLE");
	} else if (test_mode == TEST_MODE::SINGLE_MMA) {
		std::printf("%20s : %s\n", "TEST_MODE", "THREE");
	}
	float *u;
	float *h;

	cudaMallocHost(&u, sizeof(float) * N);
	cudaMallocHost(&h, sizeof(float) * N * N);

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<double> max_error_list;

	for (unsigned s = 0; s < NUM_SAMPLES; s++) {
		for (unsigned i = 0; i < N; i++) {
			u[i] = dist(mt);
		}

		cudaDeviceSynchronize();
		if (test_mode == TEST_MODE::THREE_MMA) {
			load_vector_kernel<<<1, 32>>>(h, u);
		} else if (test_mode == TEST_MODE::SINGLE_MMA) {
			load_vector_with_error_kernel<<<1, 32>>>(h, u);
		}
		cudaDeviceSynchronize();

		double max_error = 0.0;
		for (unsigned i = 0; i < N; i++) {
			for (unsigned j = 0; j < N; j++) {
				const double diff = static_cast<double>(u[i]) * static_cast<double>(u[j]) - static_cast<double>(h[i * N + j]);
				max_error = std::max(max_error, std::abs(diff));
			}
		}
		max_error_list.push_back(max_error);
	}

	const auto mean = std::accumulate(max_error_list.begin(), max_error_list.end(), 0.0) / NUM_SAMPLES;
	const auto var = std::accumulate(max_error_list.begin(), max_error_list.end(), 0.0, [&mean](const double a, const double b){return (b - mean) * (b - mean) + a;}) / NUM_SAMPLES;

	std::printf("%20s : %u\n", "N", N);
	std::printf("%20s : %u\n", "#samples", NUM_SAMPLES);
	std::printf("%20s : %e\n", "mean", mean);
	std::printf("%20s : %e\n", "var", var);

	cudaFreeHost(u);
	cudaFreeHost(h);
}

int main() {
	test(TEST_MODE::SINGLE_MMA);
	test(TEST_MODE::THREE_MMA);
}

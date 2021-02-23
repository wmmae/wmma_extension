#include <iostream>
#include <chrono>
#include <type_traits>
#include <mma.h>
#include <wmma_extension/wmma_extension.hpp>

constexpr std::size_t block_size = 256;
constexpr unsigned warp_size = 32;

#ifndef CUDA_ARCH_SM
#define CUDA_ARCH_SM 0
#endif

template <bool UseWMMAe>
__global__ void direct_product(float* const c_ptr, const half* const a_ptr, const half* const b_ptr, unsigned dim);

template <>
__global__ void direct_product<true>(float* const c_ptr, const half* const a_ptr, const half* const b_ptr, unsigned dim) {
	constexpr unsigned FDIM = 16;
	__shared__ half A_smem[warp_size];
	__shared__ half B_smem[block_size];
	__shared__ float C_smem[warp_size * block_size];

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned unique_id = threadIdx.x & 0x1f;

	float* const C_smem_ptr = C_smem + warp_size * warp_size * warp_id;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> A_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> B_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> C_frag[4];

	// load A
	const unsigned a_start = blockIdx.x * warp_size;
	if (warp_id == 0) {
		A_smem[unique_id] = a_ptr[a_start + unique_id];
	}
	__syncthreads();
	mtk::wmma::load_vector(A_frag[0], A_smem);
	mtk::wmma::load_vector(A_frag[1], A_smem + FDIM);
	nvcuda::wmma::fill_fragment(B_frag[0], __float2half(0.0f));
	nvcuda::wmma::fill_fragment(B_frag[1], __float2half(0.0f));

	for (unsigned b_start = 0; b_start < dim; b_start += block_size) {
		nvcuda::wmma::fill_fragment(C_frag[0], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[1], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[2], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[3], __float2half(0.0f));
		// load B
		B_smem[threadIdx.x] = __ldg(&b_ptr[b_start + threadIdx.x]);

		mtk::wmma::load_vector(B_frag[0], B_smem + warp_size * warp_id, false);
		nvcuda::wmma::mma_sync(C_frag[0], A_frag[0], B_frag[0], C_frag[0]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr, C_frag[0], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::mma_sync(C_frag[1], A_frag[1], B_frag[0], C_frag[1]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM, C_frag[1], warp_size, nvcuda::wmma::mem_col_major);

		mtk::wmma::load_vector(B_frag[1], B_smem + warp_size * warp_id + FDIM, false);
		nvcuda::wmma::mma_sync(C_frag[2], A_frag[0], B_frag[1], C_frag[2]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + warp_size * FDIM, C_frag[2], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::mma_sync(C_frag[3], A_frag[1], B_frag[1], C_frag[3]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM + warp_size * FDIM, C_frag[3], warp_size, nvcuda::wmma::mem_col_major);

		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			c_ptr[a_start + unique_id + (warp_id + i / warp_size + b_start) * dim] = C_smem[i + threadIdx.x];
		}
	}
}

template <>
__global__ void direct_product<false>(float* const c_ptr, const half* const a_ptr, const half* const b_ptr, unsigned dim) {
	constexpr unsigned FDIM = 16;
	__shared__ half B_smem[block_size * FDIM];
	half* const A_smem = B_smem;
	__shared__ float C_smem[warp_size * block_size];

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned unique_id = threadIdx.x & 0x1f;
	const unsigned a_start = blockIdx.x * warp_size;

	float* const C_smem_ptr = C_smem + warp_size * warp_size * warp_id;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> A_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> B_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> C_frag[4];

	// load A
	for (unsigned i = 0; i < FDIM * warp_size; i += block_size) {
		A_smem[i + threadIdx.x] = __float2half(0.0f);
	}
	if (warp_id == 0) {
		A_smem[unique_id] = a_ptr[a_start + unique_id];
	}
	__syncthreads();
	nvcuda::wmma::load_matrix_sync(A_frag[0], A_smem, warp_size);
	nvcuda::wmma::load_matrix_sync(A_frag[1], A_smem + FDIM, warp_size);
	// init B
	for (unsigned i = 0; i < FDIM * block_size; i += block_size) {
		B_smem[i + threadIdx.x] = __float2half(0.0f);
	}
	__syncthreads();


	for (unsigned b_start = 0; b_start < dim; b_start += block_size) {
		nvcuda::wmma::fill_fragment(C_frag[0], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[1], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[2], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[3], __float2half(0.0f));
		// load B
		B_smem[threadIdx.x] = __ldg(&b_ptr[b_start + threadIdx.x]);

		nvcuda::wmma::load_matrix_sync(B_frag[0], B_smem + warp_size * warp_id, block_size);
		nvcuda::wmma::mma_sync(C_frag[0], A_frag[0], B_frag[0], C_frag[0]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr, C_frag[0], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::mma_sync(C_frag[1], A_frag[1], B_frag[0], C_frag[1]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM, C_frag[1], warp_size, nvcuda::wmma::mem_col_major);

		nvcuda::wmma::load_matrix_sync(B_frag[1], B_smem + warp_size * warp_id + FDIM, block_size);
		nvcuda::wmma::mma_sync(C_frag[2], A_frag[0], B_frag[1], C_frag[2]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + warp_size * FDIM, C_frag[2], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::mma_sync(C_frag[3], A_frag[1], B_frag[1], C_frag[3]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM + warp_size * FDIM, C_frag[3], warp_size, nvcuda::wmma::mem_col_major);

		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			c_ptr[a_start + unique_id + (warp_id + i / warp_size + b_start) * dim] = C_smem[i + threadIdx.x];
		}
	}
}

template <bool UseWMMAe>
void test_direct_product(const unsigned size_power) {
	constexpr std::size_t C = 1lu << 6;
	const unsigned DIM = 1lu << size_power;
	const std::size_t grid_size = DIM / warp_size;

	half *dA, *dB;
	float *dC;
	cudaMalloc(&dA, sizeof(half) * DIM);
	cudaMalloc(&dB, sizeof(half) * DIM);
	cudaMalloc(&dC, sizeof(float) * DIM * DIM);

	half *hA;
	float *hC;
	cudaMallocHost(&hA, sizeof(half) * DIM);
	cudaMallocHost(&hC, sizeof(float) * DIM * DIM);
	for (unsigned i = 0; i < DIM; i++) hA[i] = __float2half(static_cast<float>(i) / DIM);
	cudaMemcpy(dA, hA, sizeof(half) * DIM, cudaMemcpyDefault);
	cudaMemcpy(dB, hA, sizeof(half) * DIM, cudaMemcpyDefault);

	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t c = 0; c < C; c++) {
		direct_product<UseWMMAe><<<grid_size, block_size>>>(
				dC,
				dA,
				dB,
				DIM
				);
	}
	const auto status = cudaGetLastError();
	cudaDeviceSynchronize();
	if (status != 0) {
		std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
	}
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / 1.e6 / C;

	cudaMemcpy(hC, dC, sizeof(float) * DIM * DIM, cudaMemcpyDefault);
	float diff_norm = 0.0f;
	float base_norm = 0.0f;
	for (std::size_t i = 0; i < DIM ;i++) {
		for (std::size_t j = 0; j < DIM; j++) {
			const auto corr = __half2float(hA[i]) * __half2float(hA[j]);
			const auto diff = hC[i + DIM * j] - corr;
			base_norm += corr * corr;
			diff_norm += diff * diff;
		}
	}
	std::printf("%u,%u,%u,%e,%e,%e\n",
			static_cast<unsigned>(CUDA_ARCH_SM),
			DIM,
			(UseWMMAe ? 1u : 0u),
			elapsed_time,
			(2 * DIM * DIM / elapsed_time / (1lu<<40)),
			sqrt(diff_norm / base_norm)
			);

	cudaFreeHost(hA);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void test_direct_product(const unsigned min_p, const unsigned max_p) {
	std::printf("# %s\n", __func__);
	std::printf("-- 1\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_direct_product<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_direct_product<true>(i);
	}
	std::printf("-- 2\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_direct_product<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_direct_product<true>(i);
	}
}

int main() {
	test_direct_product(8, 15);
}

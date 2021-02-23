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
__global__ void householder_16x16(float* const c_ptr, const half* const u_ptr);

template <>
__global__ void householder_16x16<true>(float* const c_ptr, const half* const u_ptr) {
	constexpr unsigned DIM = 32;
	constexpr unsigned FDIM = 16;
	__shared__ half u_smem[block_size];
	__shared__ float c_smem[block_size * DIM];

	const unsigned warp_id = threadIdx.x >> 5;
	half* const u_smem_ptr = u_smem + warp_id * DIM;
	float* const c_smem_ptr = c_ptr + warp_id * DIM * DIM;

	u_smem[threadIdx.x] = u_ptr[blockIdx.x * block_size + threadIdx.x];

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> a_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> b_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> c_frag[4];

	mtk::wmma::load_vector(a_frag[0], u_smem_ptr);
	mtk::wmma::load_vector(a_frag[1], u_smem_ptr + FDIM);
	mtk::wmma::load_vector(b_frag[0], u_smem_ptr);
	mtk::wmma::load_vector(b_frag[1], u_smem_ptr + FDIM);

	mtk::wmma::make_identity_matrix(c_frag[0]);
	nvcuda::wmma::mma_sync(c_frag[0], a_frag[0], b_frag[0], c_frag[0]);
	nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[1], a_frag[1], b_frag[0], c_frag[1]);
	nvcuda::wmma::fill_fragment(c_frag[2], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[2], a_frag[0], b_frag[1], c_frag[2]);
	mtk::wmma::make_identity_matrix(c_frag[3]);
	nvcuda::wmma::mma_sync(c_frag[3], a_frag[1], b_frag[1], c_frag[3]);

	nvcuda::wmma::store_matrix_sync(c_smem_ptr, c_frag[0], DIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(c_smem_ptr + FDIM, c_frag[1], DIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(c_smem_ptr + FDIM * DIM, c_frag[2], DIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(c_smem_ptr + FDIM * DIM + FDIM, c_frag[3], DIM, nvcuda::wmma::mem_col_major);

	for (unsigned i = 0; i < DIM * block_size; i+= block_size) {
		c_ptr[blockIdx.x * DIM + threadIdx.x + i] = c_smem[threadIdx.x + i];
	}
}

template <>
__global__ void householder_16x16<false>(float* const c_ptr, const half* const u_ptr) {
	constexpr unsigned DIM = 32;
	constexpr unsigned FDIM = 16;
	__shared__ half u_smem[block_size];
	__shared__ float c_smem[block_size * DIM];

	const unsigned warp_id = threadIdx.x >> 5;
	half* const u_smem_ptr = u_smem + warp_id * DIM;
	float* const c_smem_ptr = c_ptr + warp_id * DIM * DIM;

	u_smem[threadIdx.x] = u_ptr[blockIdx.x * block_size + threadIdx.x];

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> a_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> b_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> c_frag[4];

	mtk::wmma::load_vector(a_frag[0], u_smem_ptr);
	mtk::wmma::load_vector(a_frag[1], u_smem_ptr + FDIM);
	mtk::wmma::load_vector(b_frag[0], u_smem_ptr);
	mtk::wmma::load_vector(b_frag[1], u_smem_ptr + FDIM);

	nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);
	nvcuda::wmma::fill_fragment(c_frag[2], 0.0f);

	const unsigned unique_id = threadIdx.x & 0x1f;
	for (unsigned i = 0; i < FDIM * FDIM; i += warp_size) {
		if (unique_id % (FDIM + 1) == 0) {
			c_smem_ptr[unique_id] = 1.0f;
		} else {
			c_smem_ptr[unique_id] = 0.0f;
		}
	}
	__syncthreads();
	nvcuda::wmma::load_matrix_sync(c_frag[0], c_smem_ptr, FDIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::load_matrix_sync(c_frag[3], c_smem_ptr, FDIM, nvcuda::wmma::mem_col_major);

	nvcuda::wmma::mma_sync(c_frag[0], a_frag[0], b_frag[0], c_frag[0]);
	nvcuda::wmma::mma_sync(c_frag[1], a_frag[1], b_frag[0], c_frag[1]);
	nvcuda::wmma::mma_sync(c_frag[2], a_frag[0], b_frag[1], c_frag[2]);
	nvcuda::wmma::mma_sync(c_frag[3], a_frag[1], b_frag[1], c_frag[3]);

	nvcuda::wmma::store_matrix_sync(c_smem_ptr, c_frag[0], DIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(c_smem_ptr + FDIM, c_frag[1], DIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(c_smem_ptr + FDIM * DIM, c_frag[2], DIM, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(c_smem_ptr + FDIM * DIM + FDIM, c_frag[3], DIM, nvcuda::wmma::mem_col_major);

	for (unsigned i = 0; i < DIM * block_size; i+= block_size) {
		c_ptr[blockIdx.x * DIM + threadIdx.x + i] = c_smem[threadIdx.x + i];
	}
}

template <bool UseWMMAe>
void test_householder(const unsigned size_power) {
	constexpr std::size_t C = 1lu << 6;
	const unsigned batch_size = 1lu << size_power;
	const std::size_t grid_size = batch_size / (block_size / warp_size);

	half *dU;
	float *dC;
	cudaMalloc(&dU, sizeof(half) * batch_size * warp_size);
	cudaMalloc(&dC, sizeof(float) * batch_size * warp_size * warp_size);


	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t c = 0; c < C; c++) {
		householder_16x16<UseWMMAe><<<grid_size, block_size>>>(
				dC,
				dU
				);
	}
	const auto status = cudaGetLastError();
	cudaDeviceSynchronize();
	if (status != 0) {
		std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
	}
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / 1.e6 / C;

	std::printf("%u,%u,%u,%e\n",
			static_cast<unsigned>(CUDA_ARCH_SM),
			batch_size,
			(UseWMMAe ? 1u : 0u),
			elapsed_time
			);

	cudaFree(dU);
	cudaFree(dC);
}

void test_householder(const unsigned min_p, const unsigned max_p) {
	std::printf("# %s\n", __func__);
	std::printf("-- 1\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_householder<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_householder<true>(i);
	}
	std::printf("-- 2\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_householder<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_householder<true>(i);
	}
}

int main() {
	test_householder(8, 14);
}

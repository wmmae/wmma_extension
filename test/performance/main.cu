#include <iostream>
#include <chrono>
#include <type_traits>
#include <mma.h>
#include <wmma_extension.hpp>

constexpr std::size_t block_size = 256;
constexpr unsigned warp_size = 32;

#ifndef CUDA_ARCH_SM
#define CUDA_ARCH_SM 0
#endif

//#define KERNEL_BREAKDOWN

template <class T>
__device__ void copy16x16(T* const dst_ptr, const T* const src_ptr, const unsigned unique_id) {
	constexpr unsigned dim = 16;
	for (unsigned i = 0; i < dim; i+= (warp_size / dim)) {
		dst_ptr[i * dim + unique_id] = src_ptr[i * dim + unique_id];
	}
}

template <class T>
__device__ void copy16(T* const dst_ptr, const T* const src_ptr, const unsigned unique_id) {
	constexpr unsigned dim = 16;
	if (unique_id < dim) {
		dst_ptr[unique_id] = src_ptr[unique_id];
	}
}

template <class T>
__device__ void fill16x16(T* const dst_ptr, const T v, const unsigned unique_id) {
	constexpr unsigned DIM = 16;
	constexpr unsigned warp_size = 32;
	for (unsigned i = 0; i < DIM; i+= (warp_size / DIM)) {
		dst_ptr[i * DIM + unique_id] = v;
	}
}

template <bool UseWMMAe>
__global__ void batched_direct_product_16x16(float* const c_ptr, const half* const u_ptr);

template <>
__global__ void batched_direct_product_16x16<true>(float* const c_ptr, const half* const u_ptr) {
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

	mtk::wmma::load_vector_sync(a_frag[0], u_smem_ptr);
	mtk::wmma::load_vector_sync(a_frag[1], u_smem_ptr + FDIM);
	mtk::wmma::load_vector_sync(b_frag[0], u_smem_ptr);
	mtk::wmma::load_vector_sync(b_frag[1], u_smem_ptr + FDIM);

	nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[0], a_frag[0], b_frag[0], c_frag[0]);
	nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[1], a_frag[1], b_frag[0], c_frag[1]);
	nvcuda::wmma::fill_fragment(c_frag[2], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[2], a_frag[0], b_frag[1], c_frag[2]);
	nvcuda::wmma::fill_fragment(c_frag[3], 0.0f);
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
__global__ void batched_direct_product_16x16<false>(float* const c_ptr, const half* const u_ptr) {
	constexpr unsigned DIM = 32;
	constexpr unsigned FDIM = 16;
	__shared__ half u_tmp_smem[block_size / warp_size * FDIM * FDIM];
	__shared__ float c_smem[block_size * DIM];

	const unsigned warp_id = threadIdx.x >> 5;
	half* const u_smem_ptr = u_tmp_smem + warp_id * FDIM * FDIM;
	float* const c_smem_ptr = c_ptr + warp_id * DIM * DIM;

	for (std::size_t i = 0; i < FDIM * FDIM; i += warp_size) {
		u_smem_ptr[i + threadIdx.x] = __float2half(0.0f);
	}

	u_tmp_smem[threadIdx.x] = u_ptr[blockIdx.x * block_size + threadIdx.x];

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> a_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> b_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> c_frag[4];

	nvcuda::wmma::load_matrix_sync(a_frag[0], u_smem_ptr, DIM);
	nvcuda::wmma::load_matrix_sync(a_frag[1], u_smem_ptr + FDIM, DIM);
	nvcuda::wmma::load_matrix_sync(b_frag[0], u_smem_ptr, DIM);
	nvcuda::wmma::load_matrix_sync(b_frag[1], u_smem_ptr + FDIM, DIM);

	nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[0], a_frag[0], b_frag[0], c_frag[0]);
	nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[1], a_frag[1], b_frag[0], c_frag[1]);
	nvcuda::wmma::fill_fragment(c_frag[2], 0.0f);
	nvcuda::wmma::mma_sync(c_frag[2], a_frag[0], b_frag[1], c_frag[2]);
	nvcuda::wmma::fill_fragment(c_frag[3], 0.0f);
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
void test_batched_direct_product(const unsigned size_power) {
	constexpr std::size_t C = 1lu << 6;
	const unsigned batch_size = 1lu << size_power;
	const std::size_t grid_size = batch_size / (block_size / warp_size);

	half *dU;
	float *dC;
	cudaMalloc(&dU, sizeof(half) * batch_size * warp_size);
	cudaMalloc(&dC, sizeof(float) * batch_size * warp_size * warp_size);


	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t c = 0; c < C; c++) {
		batched_direct_product_16x16<UseWMMAe><<<grid_size, block_size>>>(
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

void test_batched_direct_product(const unsigned min_p, const unsigned max_p) {
	std::printf("# %s\n", __func__);
	std::printf("-- 1\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_batched_direct_product<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_batched_direct_product<true>(i);
	}
	std::printf("-- 2\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_batched_direct_product<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_batched_direct_product<true>(i);
	}
}

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
	mtk::wmma::load_vector_sync(A_frag[0], A_smem);
	mtk::wmma::load_vector_sync(A_frag[1], A_smem + FDIM);
	nvcuda::wmma::fill_fragment(B_frag[0], __float2half(0.0f));
	nvcuda::wmma::fill_fragment(B_frag[1], __float2half(0.0f));

	for (unsigned b_start = 0; b_start < dim; b_start += block_size) {
		nvcuda::wmma::fill_fragment(C_frag[0], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[1], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[2], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[3], __float2half(0.0f));
		// load B
		B_smem[threadIdx.x] = __ldg(&b_ptr[b_start + threadIdx.x]);

		mtk::wmma::load_vector_sync(B_frag[0], B_smem + warp_size * warp_id, false);
		nvcuda::wmma::mma_sync(C_frag[0], A_frag[0], B_frag[0], C_frag[0]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr, C_frag[0], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::mma_sync(C_frag[1], A_frag[1], B_frag[0], C_frag[1]);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM, C_frag[1], warp_size, nvcuda::wmma::mem_col_major);

		mtk::wmma::load_vector_sync(B_frag[1], B_smem + warp_size * warp_id + FDIM, false);
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

	mtk::wmma::load_vector_sync(a_frag[0], u_smem_ptr);
	mtk::wmma::load_vector_sync(a_frag[1], u_smem_ptr + FDIM);
	mtk::wmma::load_vector_sync(b_frag[0], u_smem_ptr);
	mtk::wmma::load_vector_sync(b_frag[1], u_smem_ptr + FDIM);

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

	mtk::wmma::load_vector_sync(a_frag[0], u_smem_ptr);
	mtk::wmma::load_vector_sync(a_frag[1], u_smem_ptr + FDIM);
	mtk::wmma::load_vector_sync(b_frag[0], u_smem_ptr);
	mtk::wmma::load_vector_sync(b_frag[1], u_smem_ptr + FDIM);

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

template <bool UseWMMAe>
__global__ void matmul(float* const c_ptr, const float* const a_ptr, const float* const b_ptr, const unsigned n);

template <>
__global__ void matmul<true>(float* const c_ptr, const float* const a_ptr, const float* const b_ptr, const unsigned n) {
	constexpr unsigned FDIM = 16;
	__shared__ float F32_smem[block_size * warp_size];
	__shared__ half F16_smem[block_size * warp_size];

	const unsigned unique_id = threadIdx.x & 0x1f;
	const unsigned warp_id = threadIdx.x >> 5;

	const unsigned block_c_row = blockIdx.x % (n / warp_size);
	const unsigned block_c_col = blockIdx.x / (n / warp_size);

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> frag_c[4];
	for (unsigned i = 0; i < 4; i++) {
		nvcuda::wmma::fill_fragment(frag_c[i], 0.0f);
	}

	for (unsigned kb = 0; kb < n; kb += block_size) {
		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			const auto v = a_ptr[block_c_row + unique_id + (warp_id + i / warp_size + kb) * n];
			const unsigned smem_index = threadIdx.x;
			F32_smem[smem_index] = v;
			F16_smem[smem_index] = __float2half(v);
		}

		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> frag_a[4], frag_da[4];
		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				unsigned long offset = warp_id * warp_size * warp_size + i * FDIM + j * FDIM * warp_size;
				nvcuda::wmma::load_matrix_sync(frag_a[i + 2 * j], F16_smem + offset, warp_size);
			}
		}
		const float* matrix_a_block_head = F32_smem + warp_id * warp_size * warp_size;
		mtk::wmma::foreach(
				frag_a[0],
				[&](const unsigned f_index, const unsigned m_index) {
					frag_da[0].x[f_index] = __float2half(matrix_a_block_head[m_index] - __half2float(frag_a[0].x[f_index]));
					frag_da[1].x[f_index] = __float2half(matrix_a_block_head[m_index + FDIM] - __half2float(frag_a[1].x[f_index]));
					frag_da[2].x[f_index] = __float2half(matrix_a_block_head[m_index + FDIM * warp_size] - __half2float(frag_a[2].x[f_index]));
					frag_da[3].x[f_index] = __float2half(matrix_a_block_head[m_index + FDIM * warp_size + FDIM] - __half2float(frag_a[3].x[f_index]));
				});

		__syncthreads();
		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			const auto v = b_ptr[(block_c_col + i / block_size) * n + kb + threadIdx.x];
			const unsigned smem_index = threadIdx.x + i;
			F32_smem[smem_index] = v;
			F16_smem[smem_index] = __float2half(v);
		}
		__syncthreads();
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> frag_b[4], frag_db[4];
		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				unsigned long offset = warp_id * warp_size + i * FDIM + j * FDIM * block_size;
				nvcuda::wmma::load_matrix_sync(frag_b[i + 2 * j], F16_smem + offset, block_size);
			}
		}
		const float* matrix_b_block_head = F32_smem + warp_id * warp_size;
		mtk::wmma::foreach(
				frag_a[0],
				[&](const unsigned f_index, const unsigned m_index) {
					const unsigned r = m_index & 0xf;
					const unsigned c = m_index >> 4;
					const unsigned long i = r + c * block_size;
					frag_db[0].x[f_index] = __float2half(matrix_b_block_head[i] - __half2float(frag_b[0].x[f_index]));
					frag_db[1].x[f_index] = __float2half(matrix_b_block_head[i + FDIM] - __half2float(frag_b[1].x[f_index]));
					frag_db[2].x[f_index] = __float2half(matrix_b_block_head[i + FDIM * block_size] - __half2float(frag_b[2].x[f_index]));
					frag_db[3].x[f_index] = __float2half(matrix_b_block_head[i + FDIM * block_size + FDIM] - __half2float(frag_b[3].x[f_index]));
				});

		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				const unsigned c_index = i + j * 2;
				for (unsigned k = 0; k < 2; k++) {
					const unsigned a_index = i + k * 2;
					const unsigned b_index = j * 2 + k;

					nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index], frag_b[b_index], frag_c[c_index]);
					nvcuda::wmma::mma_sync(frag_c[c_index], frag_da[a_index], frag_b[b_index], frag_c[c_index]);
					nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index], frag_db[b_index], frag_c[c_index]);
				}
			}
		}
	}
	for (unsigned i = 0; i < 2; i++) {
		for (unsigned j = 0; j < 2; j++) {
			const unsigned c_index = i + j * 2;
			nvcuda::wmma::store_matrix_sync(F32_smem + warp_size * warp_size * warp_id + i * FDIM + j * warp_size * FDIM, frag_c[c_index], warp_size, nvcuda::wmma::mem_col_major);
		}
	}
	__syncthreads();
	for (unsigned i = 0; i < warp_size * warp_size; i += block_size) {
		float v = 0.0f;
		for (unsigned j = 0; j < (block_size / warp_size); j++) {
			v += F32_smem[i + threadIdx.x + j * warp_size * warp_size];
		}
		c_ptr[(block_c_col + warp_id) * n + block_c_row + unique_id] = v;
	}
}

template <>
__global__ void matmul<false>(float* const c_ptr, const float* const a_ptr, const float* const b_ptr, const unsigned n) {
	constexpr unsigned FDIM = 16;
	__shared__ float F32_smem[block_size * warp_size];
	__shared__ half F16_smem[block_size * warp_size];

	const unsigned unique_id = threadIdx.x & 0x1f;
	const unsigned warp_id = threadIdx.x >> 5;

	const unsigned block_c_row = blockIdx.x % (n / warp_size);
	const unsigned block_c_col = blockIdx.x / (n / warp_size);

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> frag_c[4];
	for (unsigned i = 0; i < 4; i++) {
		nvcuda::wmma::fill_fragment(frag_c[i], 0.0f);
	}

	for (unsigned kb = 0; kb < n; kb += block_size) {
		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			const auto v = a_ptr[block_c_row + unique_id + (warp_id + i / warp_size + kb) * n];
			const unsigned smem_index = threadIdx.x;
			F32_smem[smem_index] = v;
			F16_smem[smem_index] = __float2half(v);
		}

		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> frag_a[4], frag_da[4];
		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				unsigned long offset = warp_id * warp_size * warp_size + i * FDIM + j * FDIM * warp_size;
				nvcuda::wmma::load_matrix_sync(frag_a[i + 2 * j], F16_smem + offset, warp_size);
			}
		}
		__syncthreads();
		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			const unsigned smem_index = threadIdx.x + i;
			F16_smem[smem_index] = __float2half(F32_smem[smem_index] - __half2float(F16_smem[smem_index]));
		}
		__syncthreads();
		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				unsigned long offset = warp_id * warp_size + i * FDIM + j * FDIM * block_size;
				nvcuda::wmma::load_matrix_sync(frag_da[i + 2 * j], F16_smem + offset, block_size);
			}
		}
		__syncthreads();

		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			const auto v = b_ptr[(block_c_col + i / block_size) * n + kb + threadIdx.x];
			const unsigned smem_index = threadIdx.x + i;
			F32_smem[smem_index] = v;
			F16_smem[smem_index] = __float2half(v);
		}
		__syncthreads();
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> frag_b[4], frag_db[4];
		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				unsigned long offset = warp_id * warp_size + i * FDIM + j * FDIM * block_size;
				nvcuda::wmma::load_matrix_sync(frag_b[i + 2 * j], F16_smem + offset, block_size);
			}
		}
		__syncthreads();
		for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
			const unsigned smem_index = threadIdx.x + i;
			F16_smem[smem_index] = __float2half(F32_smem[smem_index] - __half2float(F16_smem[smem_index]));
		}
		__syncthreads();
		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				unsigned long offset = warp_id * warp_size + i * FDIM + j * FDIM * block_size;
				nvcuda::wmma::load_matrix_sync(frag_db[i + 2 * j], F16_smem + offset, block_size);
			}
		}

		for (unsigned i = 0; i < 2; i++) {
			for (unsigned j = 0; j < 2; j++) {
				const unsigned c_index = i + j * 2;
				for (unsigned k = 0; k < 2; k++) {
					const unsigned a_index = i + k * 2;
					const unsigned b_index = j * 2 + k;

					nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index], frag_b[b_index], frag_c[c_index]);
					nvcuda::wmma::mma_sync(frag_c[c_index], frag_da[a_index], frag_b[b_index], frag_c[c_index]);
					nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index], frag_db[b_index], frag_c[c_index]);
				}
			}
		}
	}
	for (unsigned i = 0; i < 2; i++) {
		for (unsigned j = 0; j < 2; j++) {
			const unsigned c_index = i + j * 2;
			nvcuda::wmma::store_matrix_sync(F32_smem + warp_size * warp_size * warp_id + i * FDIM + j * warp_size * FDIM, frag_c[c_index], warp_size, nvcuda::wmma::mem_col_major);
		}
	}
	__syncthreads();
	for (unsigned i = 0; i < warp_size * warp_size; i += block_size) {
		float v = 0.0f;
		for (unsigned j = 0; j < (block_size / warp_size); j++) {
			v += F32_smem[i + threadIdx.x + j * warp_size * warp_size];
		}
		c_ptr[(block_c_col + warp_id) * n + block_c_row + unique_id] = v;
	}
}

template <bool UseWMMAe>
void test_matmul(const unsigned size_power) {
	constexpr std::size_t C = 1lu << 6;
	const std::size_t N = 1lu << size_power;
	const std::size_t grid_size = (N / warp_size) * (N / warp_size);

	float *dA, *dB, *dC;
	cudaMalloc(&dA, sizeof(float) * N * N);
	cudaMalloc(&dB, sizeof(float) * N * N);
	cudaMalloc(&dC, sizeof(float) * N * N);


	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t c = 0; c < C; c++) {
		matmul<UseWMMAe><<<grid_size, block_size>>>(
				dC,
				dA,
				dB,
				N
				);
	}
	const auto status = cudaGetLastError();
	cudaDeviceSynchronize();
	if (status != 0) {
		std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
	}
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / 1.e6 / C;

	std::printf("%u,%lu,%u,%e,%e\n",
			static_cast<unsigned>(CUDA_ARCH_SM),
			N,
			(UseWMMAe ? 1u : 0u),
			elapsed_time,
			(2 * N * N * N) / elapsed_time / (1lu << 40)
			);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void test_matmul(const unsigned min_p, const unsigned max_p) {
	std::printf("# %s\n", __func__);
	std::printf("-- 1\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_matmul<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_matmul<true>(i);
	}
	std::printf("-- 2\n");
	for (unsigned i = min_p; i <= max_p; i++) {
		test_matmul<false>(i);
	}
	for (unsigned i = min_p; i <= max_p; i++) {
		test_matmul<true>(i);
	}
}

int main() {
	test_batched_direct_product(8, 15);
	test_householder(8, 14);
	test_direct_product(8, 15);
	test_matmul(8, 14);
}

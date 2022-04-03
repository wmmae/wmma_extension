#include <iostream>
#include <chrono>
#include <cassert>
#include <cublas.h>
#include <cublas_v2.h>
#include "utils.hpp"

namespace {
constexpr unsigned warp_size = 32;
constexpr uint32_t smem_skew = 8;

__device__ uint32_t get_smem_ptr_uint(const void* const ptr) {
  uint32_t smem_ptr;

  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));

  return smem_ptr;
}

template <unsigned SizeInBytes>
__device__ inline void cp_async(void* const smem, const void* const gmem) {
	const unsigned smem_int_ptr = get_smem_ptr_uint(smem);
	asm volatile(
          "{\n"
          "cp.async.ca.shared.global [%0], [%1], %2;\n"
          "}\n" ::
          "r"(smem_int_ptr), "l"(gmem), "n"(SizeInBytes));
}

__device__ inline void cp_async_commit() {
	asm volatile(
          "{\n"
		  "cp.async.commit_group;\n"
          "}\n");
}

__device__ inline void cp_async_wait_all() {
	asm volatile(
          "{\n"
		  "cp.async.wait_all;\n"
          "}\n");
}

template <int N>
__device__ inline void cp_async_wait_group() {
	asm volatile(
          "{\n"
		  "cp.async.wait_group %0;\n"
          "}\n":: "n"(N));
}

// SMEM_M * SMEM_N must be larger than or equal to BLOCK_SIZE
template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
__device__ void dmem2smem(
		float* const dst_smem,
		const unsigned m, const unsigned n,
		const float* const src_dmem, const unsigned ld
		) {
	if (m == SMEM_M && n == SMEM_N) {
		if ((SMEM_M & 0b11) == 0) {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE * 4) {
				const auto j = i + threadIdx.x * 4;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;
				const auto dmem_index = j_m + j_n * ld;
				const auto smem_index = j_m + j_n * (SMEM_M + smem_skew);

				cp_async<4 * 4>(&dst_smem[smem_index], &src_dmem[dmem_index]);
			}
		} else {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;
				const auto dmem_index = j_m + j_n * ld;
				const auto smem_index = j_m + j_n * (SMEM_M + smem_skew);

				dst_smem[smem_index] = src_dmem[dmem_index];
			}
		}
	} else {
		for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
			const auto j = i + threadIdx.x;
			const auto j_m = j % SMEM_M;
			const auto j_n = j / SMEM_M;
			const auto dmem_index = j_m + j_n * ld;
			const auto smem_index = j_m + j_n * (SMEM_M + smem_skew);

			float v = 0.f;
			if (j_m < m && j_n < n) {
				v = src_dmem[dmem_index];
			}

			dst_smem[smem_index] = v;
		}
	}
}

// SMEM_M * SMEM_N must be larger than or equal to BLOCK_SIZE
template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
__device__ void smem2dmem(
		float* const dst_dmem, const unsigned ld,
		const unsigned m, const unsigned n,
		const float* const src_smem,
		const float alpha, const float beta
		) {
	if (beta == 0.f) {
		if (m == SMEM_M && n == SMEM_N) {
			if ((SMEM_M & 0b11) == 0) {
				for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE * 4) {
					const auto j = i + threadIdx.x * 4;
					const auto j_m = j % SMEM_M;
					const auto j_n = j / SMEM_M;
					const auto mem_index = j_m + j_n * ld;

					auto tmp_v4 = make_float4(
							src_smem[j + 0] * alpha,
							src_smem[j + 1] * alpha,
							src_smem[j + 2] * alpha,
							src_smem[j + 3] * alpha
							);

					*reinterpret_cast<float4*>(&dst_dmem[mem_index]) = tmp_v4;
				}
			} else {
				for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
					const auto j = i + threadIdx.x;
					const auto j_m = j % SMEM_M;
					const auto j_n = j / SMEM_M;

					dst_dmem[j_m + j_n * ld] = alpha * src_smem[j];
				}
			}
		} else {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				if (j_m < m && j_n < n) {
					dst_dmem[j_m + j_n * ld] = alpha * src_smem[j];
				}
			}
		}
	} else {
		// beta is not zero
		if (m == SMEM_M && n == SMEM_N) {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				const auto dmem_offset = j_m + j_n * ld;
				dst_dmem[dmem_offset] = alpha * src_smem[j] + beta * dst_dmem[dmem_offset];
			}
		} else {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				if (j_m < m && j_n < n) {
					const auto dmem_offset = j_m + j_n * ld;
					dst_dmem[dmem_offset] = alpha * src_smem[j] + beta * dst_dmem[dmem_offset];
				}
			}
		}
	}
}

// SMEM_M * SMEM_N must be larger than or equal to BLOCK_SIZE
template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
__device__ void fill_zero(
		float* const dst_smem
		) {
	for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE * 4) {
		const auto j = i + threadIdx.x * 4;
		*reinterpret_cast<float4*>(dst_smem + j) = make_float4(0, 0, 0, 0);
	}
}

template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE,
	class FRAGMENT_T,
	class TC_Policy>
__device__ void mma_core(
		float* const c_smem,
		float* const a_smem,
		float* const b_smem
		) {
#pragma unroll
	for (unsigned w = 0; w < (SMEM_M * SMEM_N / (WARP_M * WARP_N)); w += BLOCK_SIZE / warp_size) {
		const auto wi = w + threadIdx.x / warp_size;

		constexpr unsigned num_stages = 2;

		// Load A
		mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, WARP_M, WARP_N, WARP_K, FRAGMENT_T, nvcuda::wmma::row_major, TC_Policy> frag_a[num_stages];
		const auto wi_m = (wi % (SMEM_M / WARP_M)) * WARP_M;
		const auto a_smem_offset = wi_m * (SMEM_K + smem_skew) + 0;
		mtk::wmma::tcec::load_matrix_sync(frag_a[0], a_smem + a_smem_offset, SMEM_K + smem_skew, false);

		// Load B
		mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b, WARP_M, WARP_N, WARP_K, FRAGMENT_T, nvcuda::wmma::col_major, TC_Policy> frag_b[num_stages];
		const auto wi_n = (wi / (SMEM_M / WARP_M)) * WARP_N;
		const auto b_smem_offset = wi_n * (SMEM_K + smem_skew) + 0;
		mtk::wmma::tcec::load_matrix_sync(frag_b[0], b_smem + b_smem_offset, SMEM_K + smem_skew, false);

		// Load C
		mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, WARP_M, WARP_N, WARP_K, FRAGMENT_T, void, TC_Policy> frag_c;
		const auto c_smem_offset = wi_m + wi_n * SMEM_M;
		mtk::wmma::tcec::load_matrix_sync<nvcuda::wmma::col_major>(frag_c, c_smem + c_smem_offset, SMEM_M, false);

		unsigned stage = 1;
#pragma unroll
		for (unsigned wi_k = WARP_K; wi_k < SMEM_K; wi_k += WARP_K) {

			// Load A
			const auto a_smem_offset = wi_m * (SMEM_K + smem_skew) + wi_k;
			mtk::wmma::tcec::load_matrix_sync(frag_a[stage], a_smem + a_smem_offset, SMEM_K + smem_skew, false);

			// Load B
			const auto b_smem_offset = wi_n * (SMEM_K + smem_skew) + wi_k;
			mtk::wmma::tcec::load_matrix_sync(frag_b[stage], b_smem + b_smem_offset, SMEM_K + smem_skew, false);

			stage = 1 - stage;

			// mma
			mtk::wmma::tcec::mma_sync(frag_c, frag_a[stage], frag_b[stage], frag_c);
		}
		stage = 1 - stage;
		// mma
		mtk::wmma::tcec::mma_sync(frag_c, frag_a[stage], frag_b[stage], frag_c);
		mtk::wmma::tcec::store_matrix_sync<nvcuda::wmma::col_major>(c_smem + c_smem_offset, frag_c, SMEM_M, false);
	}
}

// This kernel function computes batched matrix-matrix multiplication
// A needs to be row major, and B needst to be col major
template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE,
	unsigned BLOCK_M_PER_MATRIX,
	unsigned BLOCK_N_PER_MATRIX,
	class FRAGMENT_T,
	class TC_Policy>
__global__ void bgemm_kernel(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const float alpha,
		const float* const* const a_ptr, const unsigned lda,
		const float* const* const b_ptr, const unsigned ldb,
		const float beta,
		float* const* const c_ptr, const unsigned ldc
		) {
	constexpr unsigned num_stages = 2;
	// Sharedm memory
	extern __shared__ float smem[];

#pragma unroll 4
	for (unsigned bn = blockIdx.z * n / BLOCK_N_PER_MATRIX; bn < (blockIdx.z + 1) * n / BLOCK_N_PER_MATRIX; bn += SMEM_N) {
#pragma unroll 4
		for (unsigned bm = blockIdx.y * m / BLOCK_M_PER_MATRIX; bm < (blockIdx.y + 1) * m / BLOCK_M_PER_MATRIX; bm += SMEM_M) {
			constexpr unsigned bk = 0;

			// Load A from device memory to shared memory
			const auto real_bm = min(SMEM_M, (blockIdx.y + 1) * m / BLOCK_M_PER_MATRIX - bm);
			const auto real_bk = min(SMEM_K, k - bk);
			const auto a_dmem_offset = bm * lda + bk;
			// Device memory A
			const float* const a_dmem = a_ptr[blockIdx.x];
			// Shared memory A
			float* const a_smem = smem;
			// Load row major A using a loader for col major
			dmem2smem<SMEM_K, SMEM_M, BLOCK_SIZE>(a_smem, real_bk, real_bm, a_dmem + a_dmem_offset, lda);

			cp_async_commit();

			// Load B from global memory to shared memory
			const auto real_bn = min(SMEM_N, (blockIdx.z + 1) * n / BLOCK_N_PER_MATRIX - bn);
			const auto b_dmem_offset = bn * ldb + bk;
			// Device memory B
			const float* const b_dmem = b_ptr[blockIdx.x];
			// Shared memory B
			float* const b_smem = a_smem + SMEM_M * (SMEM_K + smem_skew) * num_stages;
			// Load col major A using a loader for col major
			dmem2smem<SMEM_K, SMEM_N, BLOCK_SIZE>(b_smem, real_bk, real_bn, b_dmem + b_dmem_offset, ldb);

			cp_async_commit();

			// Initialize C
			float* const c_smem = b_smem + (SMEM_K + smem_skew) * SMEM_N * num_stages;
			fill_zero<SMEM_M, SMEM_N, BLOCK_SIZE>(c_smem);

			unsigned stage = 0;
			for (unsigned bk = SMEM_K; bk < k; bk += SMEM_K) {
				stage = 1 - stage;

				// Load A from device memory to shared memory
				const auto a_dmem_offset = bm * lda + bk;
				dmem2smem<SMEM_K, SMEM_M, BLOCK_SIZE>(a_smem + stage * SMEM_M * (SMEM_K + smem_skew), real_bk, real_bm, a_dmem + a_dmem_offset, lda);

				cp_async_commit();

				// Load B from global memory to shared memory
				const auto b_dmem_offset = bn * ldb + bk;
				dmem2smem<SMEM_K, SMEM_N, BLOCK_SIZE>(b_smem + stage * (SMEM_K + smem_skew) * SMEM_N, real_bk, real_bn, b_dmem + b_dmem_offset, ldb);

				cp_async_commit();

				// MMA
				cp_async_wait_group<2>();
				__syncthreads();
				mma_core<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(c_smem, a_smem + (1 - stage) * SMEM_M * (SMEM_K + smem_skew), b_smem + (1 - stage) * (SMEM_K + smem_skew) * SMEM_N);
			} // loop bk

			// MMA
			cp_async_wait_all();
			__syncthreads();
			mma_core<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(c_smem, a_smem + stage * SMEM_M * (SMEM_K + smem_skew), b_smem + stage * (SMEM_K + smem_skew) * SMEM_N);
			__syncthreads();

			const auto c_dmem_offset = bm + bn * ldc;
			float* const c_dmem = c_ptr[blockIdx.x];
			smem2dmem<SMEM_M, SMEM_N, BLOCK_SIZE>(c_dmem + c_dmem_offset, ldc, real_bm, real_bn, c_smem, alpha, beta);
		} // loop bn
	} // loop bm
}

template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE,
	unsigned BLOCK_M_PER_MATRIX,
	unsigned BLOCK_N_PER_MATRIX,
	class FRAGMENT_T,
	class TC_Policy>
void bgemm(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const float alpha,
		const float* const* const a_ptr, const unsigned lda,
		const float* const* const b_ptr, const unsigned ldb,
		const float beta,
		float* const* const c_ptr, const unsigned ldc,
		const unsigned batch_size
		) {
	// Set shared memory size
	const auto shared_memory_size = ((SMEM_M * (SMEM_K + smem_skew) + SMEM_N * (SMEM_K + smem_skew)) * 2 + SMEM_M * SMEM_N) * sizeof(float);
	cudaFuncSetAttribute(&(bgemm_kernel<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, BLOCK_M_PER_MATRIX, BLOCK_N_PER_MATRIX, FRAGMENT_T, TC_Policy>), cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

	// Launch
	const dim3 grid_size(batch_size, BLOCK_M_PER_MATRIX, BLOCK_N_PER_MATRIX);
	bgemm_kernel<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, BLOCK_M_PER_MATRIX, BLOCK_N_PER_MATRIX, FRAGMENT_T, TC_Policy><<<grid_size, BLOCK_SIZE, shared_memory_size>>>(
			m, n, k,
			alpha,
			a_ptr, lda,
			b_ptr, ldb,
			beta,
			c_ptr, ldc
			);
}

template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE,
	unsigned BLOCK_M_PER_MATRIX,
	unsigned BLOCK_N_PER_MATRIX
>
void test_batched_sgemm(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const unsigned batch_size
		) {
	static_assert(SMEM_M * SMEM_N >= BLOCK_SIZE);
	static_assert(SMEM_M * SMEM_K >= BLOCK_SIZE);
	static_assert(SMEM_K * SMEM_N >= BLOCK_SIZE);
	std::printf("!-- %s\n", __func__);
	std::printf("%15s: (%u, %u, %u)\n", "Size", m, n, k);
	std::printf("%15s: (%u, %u, %u)\n", "Smem size", SMEM_M, SMEM_N, SMEM_K);
	std::printf("%15s: (%u, %u, %u)\n", "Warp size", WARP_M, WARP_N, WARP_K);
	std::printf("%15s: (%u, %u)\n", "Grid YZ", BLOCK_M_PER_MATRIX, BLOCK_N_PER_MATRIX);
	std::printf("%15s: %u\n", "Block size", BLOCK_SIZE);
	std::printf("%15s: %u\n", "Batch size", batch_size);
	std::printf("%15s: %e GiB\n", "Memory", static_cast<double>(1lu * (m * n + n * k + k * m) * batch_size * sizeof(float)) / (1lu << 30));
	std::printf("%15s: %lu byte\n", "Shared memory", ((SMEM_M * (SMEM_K + smem_skew) + SMEM_N * (SMEM_K + smem_skew)) * 2 + SMEM_M * SMEM_N) * sizeof(float));
	std::fflush(stdout);

	using FRAGMENT_T = half;
	using TC_Policy = mtk::wmma::tcec::detail::default_policy<FRAGMENT_T, mtk::wmma::tcec::with_ec, mtk::wmma::tcec::op_mma>::type;

	float **d_a_ptr_array;
	float **d_b_ptr_array;
	float **d_c_ptr_array;
	WMMAE_CUDA_CHECK_ERROR(cudaMalloc(&d_a_ptr_array, sizeof(float*) * batch_size));
	WMMAE_CUDA_CHECK_ERROR(cudaMalloc(&d_b_ptr_array, sizeof(float*) * batch_size));
	WMMAE_CUDA_CHECK_ERROR(cudaMalloc(&d_c_ptr_array, sizeof(float*) * batch_size));

	float **h_a_ptr_array;
	float **h_b_ptr_array;
	float **h_c_ptr_array;
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&h_a_ptr_array, sizeof(float*) * batch_size));
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&h_b_ptr_array, sizeof(float*) * batch_size));
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&h_c_ptr_array, sizeof(float*) * batch_size));

	// Host memory for initializing
	float* init_matrix;
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&init_matrix, sizeof(float) * m * n * k / (std::min(m, std::min(n, k)))));
	for (unsigned i = 0; i < batch_size; i++) {
		// Allocate device memory and set
		float *d_a_ptr;
		float *d_b_ptr;
		float *d_c_ptr;
		WMMAE_CUDA_CHECK_ERROR(cudaMalloc(&d_a_ptr, sizeof(float) * m * k));
		WMMAE_CUDA_CHECK_ERROR(cudaMalloc(&d_b_ptr, sizeof(float) * k * n));
		WMMAE_CUDA_CHECK_ERROR(cudaMalloc(&d_c_ptr, sizeof(float) * m * n));
		h_a_ptr_array[i] = d_a_ptr;
		h_b_ptr_array[i] = d_b_ptr;
		h_c_ptr_array[i] = d_c_ptr;

		// Initialize matrices
		// A
		for (unsigned j = 0; j < m * k; j++) init_matrix[j] = j / static_cast<float>(k);
		WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(d_a_ptr, init_matrix, sizeof(float) * m * k, cudaMemcpyDefault));
		// B
		for (unsigned j = 0; j < k * n; j++) init_matrix[j] = j / static_cast<float>(n);
		WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(d_b_ptr, init_matrix, sizeof(float) * k * n, cudaMemcpyDefault));
		// C
		for (unsigned j = 0; j < m * n; j++) init_matrix[j] = 0.f;
		WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(d_c_ptr, init_matrix, sizeof(float) * m * n, cudaMemcpyDefault));
	}
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(init_matrix));

	// Copy the pointer array to the device
	WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(d_a_ptr_array, h_a_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault));
	WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(d_b_ptr_array, h_b_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault));
	WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(d_c_ptr_array, h_c_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault));

	WMMAE_CUDA_CHECK_ERROR(cudaDeviceSynchronize());
	bgemm<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, BLOCK_M_PER_MATRIX, BLOCK_N_PER_MATRIX, FRAGMENT_T, TC_Policy>(
			m, n, k,
			1.f,
			d_a_ptr_array, k,
			d_b_ptr_array, k,
			0.f,
			d_c_ptr_array, m,
			batch_size
			);
	WMMAE_CUDA_CHECK_ERROR(cudaDeviceSynchronize());

	// evaluate the last batch matrix
	float* last_a_ptr;
	float* last_b_ptr;
	float* last_c_ptr;
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&last_a_ptr, sizeof(float) * m * k));
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&last_b_ptr, sizeof(float) * k * n));
	WMMAE_CUDA_CHECK_ERROR(cudaMallocHost(&last_c_ptr, sizeof(float) * m * n));
	WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(last_a_ptr, h_a_ptr_array[batch_size - 1], sizeof(float) * m * k, cudaMemcpyDefault));
	WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(last_b_ptr, h_b_ptr_array[batch_size - 1], sizeof(float) * k * n, cudaMemcpyDefault));
	WMMAE_CUDA_CHECK_ERROR(cudaMemcpy(last_c_ptr, h_c_ptr_array[batch_size - 1], sizeof(float) * m * n, cudaMemcpyDefault));
	double base_norm = 0.;
	double diff_norm = 0.;
#pragma omp parallel for collapse(2) reduction(+: base_norm) reduction(+: diff_norm)
	for (unsigned i = 0; i < m; i++) {
		for (unsigned j = 0; j < n; j++) {
			double c = 0.;
			for (unsigned l = 0; l < k; l++) {
				c += static_cast<double>(last_a_ptr[l + i * k]) * static_cast<double>(last_b_ptr[l + j * k]);
			}
			const auto diff = last_c_ptr[i + j * m] - c;
			const auto base = c;
			base_norm += base * base;
			diff_norm += diff * diff;
		}
	}
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(last_a_ptr));
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(last_b_ptr));
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(last_c_ptr));

	WMMAE_CUDA_CHECK_ERROR(cudaDeviceSynchronize());
	// evaluation of computing performance
	constexpr unsigned test_count = 1lu << 8;

	{
		WMMAE_CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		// evaluation of computing performance
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned c = 0; c < test_count; c++) {
		bgemm<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, BLOCK_M_PER_MATRIX, BLOCK_N_PER_MATRIX, FRAGMENT_T, TC_Policy>(
					m, n, k,
					1.f,
					d_a_ptr_array, k,
					d_b_ptr_array, k,
					0.f,
					d_c_ptr_array, m,
					batch_size
					);
		}
		WMMAE_CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		const auto end_clock = std::chrono::system_clock::now();
		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;
		const auto complexity = 2lu * static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * static_cast<std::size_t>(batch_size);
		const auto performance = complexity / elapsed_time / (1e12);

		std::printf("%15s: %e s\n", "Time", elapsed_time);
		std::printf("%15s: %e TFlop/s\n", "Performance", performance);
		std::printf("%15s: %e\n", "Error", std::sqrt(diff_norm / base_norm));
	}
	// cuBLAS
	{
		cublasHandle_t cublas_handle;
		cublasCreate(&cublas_handle);
		const float alpha = 1.f;
		const float beta = 0.f;
		cudaDeviceSynchronize();
		// evaluation of computing performance
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned c = 0; c < test_count; c++) {
			cublasSgemmBatched(cublas_handle,
					CUBLAS_OP_T, CUBLAS_OP_N,
					m, n, k,
					&alpha,
					d_a_ptr_array, k,
					d_b_ptr_array, k,
					&beta,
					d_c_ptr_array, m,
					batch_size
					);
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;
		const auto complexity = 2lu * static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * static_cast<std::size_t>(batch_size);
		const auto performance = complexity / elapsed_time / (1e12);

		std::printf("%15s: %e s (cuBLAS)\n", "Time", elapsed_time);
		std::printf("%15s: %e TFlop/s (cuBLAS)\n", "Performance", performance);
		cublasDestroy(cublas_handle);
	}

	// Free
	for (unsigned i = 0; i < batch_size; i++) {
		WMMAE_CUDA_CHECK_ERROR(cudaFree(h_a_ptr_array[i]));
		WMMAE_CUDA_CHECK_ERROR(cudaFree(h_b_ptr_array[i]));
		WMMAE_CUDA_CHECK_ERROR(cudaFree(h_c_ptr_array[i]));
	}
	WMMAE_CUDA_CHECK_ERROR(cudaFree(d_a_ptr_array));
	WMMAE_CUDA_CHECK_ERROR(cudaFree(d_b_ptr_array));
	WMMAE_CUDA_CHECK_ERROR(cudaFree(d_c_ptr_array));
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(h_a_ptr_array));
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(h_b_ptr_array));
	WMMAE_CUDA_CHECK_ERROR(cudaFreeHost(h_c_ptr_array));
}
} // noname napespace

int main() {
	constexpr unsigned m = 1024;
	constexpr unsigned n = 1024;
	constexpr unsigned k = 1024;
	constexpr unsigned batch_size = 256;
	test_batched_sgemm<64, 128, 64, 64, 32, 64, 128, 8, 8>(m, n, k, batch_size);
}

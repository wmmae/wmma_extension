#include <iostream>
#include <chrono>
#include <cublas.h>
#include <cublas_v2.h>
#include "utils.hpp"

namespace {
constexpr unsigned warp_size = 32;

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
				const auto mem_index = j_m + j_n * ld;

				const auto tmp_v4 = *reinterpret_cast<const float4*>(&src_dmem[mem_index]);

				*reinterpret_cast<float4*>(&dst_smem[j]) = tmp_v4;
			}
		} else {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				dst_smem[j] = src_dmem[j_m + j_n * ld];
			}
		}
	} else {
		for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
			const auto j = i + threadIdx.x;
			const auto j_m = j % SMEM_M;
			const auto j_n = j / SMEM_M;

			float v = 0.f;
			if (j_m < m && j_n < n) {
				v = src_dmem[j_m + j_n * ld];
			}

			dst_smem[j] = v;
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
							src_smem[j + 0],
							src_smem[j + 1],
							src_smem[j + 2],
							src_smem[j + 3]
							);
					tmp_v4.x *= alpha;
					tmp_v4.y *= alpha;
					tmp_v4.z *= alpha;
					tmp_v4.w *= alpha;

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
	for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
		const auto j = i + threadIdx.x;
		dst_smem[j] = 0.f;
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
		mtk::wmma::mma_f32::fragment<nvcuda::wmma::matrix_a, WARP_M, WARP_N, WARP_K, FRAGMENT_T, nvcuda::wmma::row_major, TC_Policy> frag_a[num_stages];
		const auto wi_m = (wi % (SMEM_M / WARP_M)) * WARP_M;
		const auto a_smem_offset = wi_m * SMEM_K + 0;
		mtk::wmma::mma_f32::load_matrix_sync(frag_a[0], a_smem + a_smem_offset, SMEM_K, false);

		// Load B
		mtk::wmma::mma_f32::fragment<nvcuda::wmma::matrix_b, WARP_M, WARP_N, WARP_K, FRAGMENT_T, nvcuda::wmma::col_major, TC_Policy> frag_b[num_stages];
		const auto wi_n = (wi / (SMEM_M / WARP_M)) * WARP_N;
		const auto b_smem_offset = wi_n * SMEM_K + 0;
		mtk::wmma::mma_f32::load_matrix_sync(frag_b[0], b_smem + b_smem_offset, SMEM_K, false);

		// Load C
		mtk::wmma::mma_f32::fragment<nvcuda::wmma::accumulator, WARP_M, WARP_N, WARP_K, FRAGMENT_T, void, TC_Policy> frag_c;
		const auto c_smem_offset = wi_m + wi_n * SMEM_M;
		mtk::wmma::mma_f32::load_matrix_sync(frag_c, c_smem + c_smem_offset, SMEM_M, nvcuda::wmma::mem_col_major);

		unsigned stage = 0;
#pragma unroll
		for (unsigned wi_k = WARP_K; wi_k < SMEM_K; wi_k += WARP_K) {
			// mma
			mtk::wmma::mma_f32::mma_sync(frag_c, frag_a[stage], frag_b[stage], frag_c);

			stage = 1 - stage;

			// Load A
			const auto a_smem_offset = wi_m * SMEM_K + wi_k;
			mtk::wmma::mma_f32::load_matrix_sync(frag_a[stage], a_smem + a_smem_offset, SMEM_K, false);

			// Load B
			const auto b_smem_offset = wi_n * SMEM_K + wi_k;
			mtk::wmma::mma_f32::load_matrix_sync(frag_b[stage], b_smem + b_smem_offset, SMEM_K, false);
		}
		// mma
		mtk::wmma::mma_f32::mma_sync(frag_c, frag_a[stage], frag_b[stage], frag_c);
		mtk::wmma::mma_f32::store_matrix_sync(c_smem + c_smem_offset, frag_c, SMEM_M, nvcuda::wmma::mem_col_major, false);
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

	for (unsigned bm = 0; bm < m; bm += SMEM_M) {
		for (unsigned bn = 0; bn < n; bn += SMEM_N) {
			constexpr unsigned bk = 0;
			// Load A from device memory to shared memory
			const auto real_bm = min(SMEM_M, m - bm);
			const auto real_bk = min(SMEM_K, k - bk);
			const auto a_dmem_offset = bm * lda + bk;
			// Device memory A
			const float* const a_dmem = a_ptr[blockIdx.x];
			// Shared memory A
			float* const a_smem = smem;
			// Load row major A using a loader for col major
			dmem2smem<SMEM_K, SMEM_M, BLOCK_SIZE>(a_smem, real_bk, real_bm, a_dmem + a_dmem_offset, lda);

			// Load B from global memory to shared memory
			const auto real_bn = min(SMEM_N, n - bn);
			const auto b_dmem_offset = bn * ldb + bk;
			// Device memory B
			const float* const b_dmem = b_ptr[blockIdx.x];
			// Shared memory B
			float* const b_smem = a_smem + SMEM_M * SMEM_K * num_stages;
			// Load col major A using a loader for col major
			dmem2smem<SMEM_K, SMEM_N, BLOCK_SIZE>(b_smem, real_bk, real_bn, b_dmem + b_dmem_offset, ldb);

			// Initialize C
			float* const c_smem = b_smem + SMEM_K * SMEM_N * num_stages;
			fill_zero<SMEM_M, SMEM_N, BLOCK_SIZE>(c_smem);

			__syncthreads();

			unsigned stage = 0;
			for (unsigned bk = SMEM_K; bk < k; bk += SMEM_K) {
				// MMA
				mma_core<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(c_smem, a_smem + stage * SMEM_M * SMEM_K, b_smem + stage * SMEM_K * SMEM_N);

				stage = 1 - stage;

				// Load A from device memory to shared memory
				const auto real_bm = min(SMEM_M, m - bm);
				const auto real_bk = min(SMEM_K, k - bk);
				const auto a_dmem_offset = bm * lda + bk;
				dmem2smem<SMEM_K, SMEM_M, BLOCK_SIZE>(a_smem + stage * SMEM_M * SMEM_K, real_bk, real_bm, a_dmem + a_dmem_offset, lda);

				// Load B from global memory to shared memory
				const auto real_bn = min(SMEM_N, n - bn);
				const auto b_dmem_offset = bn * ldb + bk;
				dmem2smem<SMEM_K, SMEM_N, BLOCK_SIZE>(b_smem + stage * SMEM_K * SMEM_N, real_bk, real_bn, b_dmem + b_dmem_offset, ldb);

				__syncthreads();
			} // loop bk

			// MMA
			mma_core<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(c_smem, a_smem + stage * SMEM_M * SMEM_K, b_smem + stage * SMEM_K * SMEM_N);

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
	const auto shared_memory_size = ((SMEM_M * SMEM_K + SMEM_K * SMEM_N) * 2 + SMEM_M * SMEM_N) * sizeof(float);
	cudaFuncSetAttribute(&(bgemm_kernel<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>), cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

	// Launch
	bgemm_kernel<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy><<<batch_size, BLOCK_SIZE, shared_memory_size>>>(
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
	unsigned BLOCK_SIZE>
void test_batched_sgemm(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const unsigned batch_size
		) {
	std::printf("!-- %s\n", __func__);
	std::printf("-------\n");
	std::printf("%15s: (%u, %u, %u)\n", "Size", m, n, k);
	std::printf("%15s: %u\n", "Batch size", batch_size);
	std::printf("%15s: %e GiB\n", "Memory", static_cast<double>(1lu * (m * n + n * k + k * m) * batch_size * sizeof(float)) / (1lu << 30));
	std::printf("%15s: %lu byte\n", "Shared memory", sizeof(float) * (SMEM_M * SMEM_K + SMEM_K * SMEM_N + SMEM_M * SMEM_N));
	std::fflush(stdout);

	using FRAGMENT_T = half;
	using TC_Policy = mtk::wmma::mma_f32::detail::default_policy<FRAGMENT_T, mtk::wmma::mma_f32::with_ec, mtk::wmma::mma_f32::op_mma>::type;

	float **d_a_ptr_array;
	float **d_b_ptr_array;
	float **d_c_ptr_array;
	cudaMalloc(&d_a_ptr_array, sizeof(float*) * batch_size);
	cudaMalloc(&d_b_ptr_array, sizeof(float*) * batch_size);
	cudaMalloc(&d_c_ptr_array, sizeof(float*) * batch_size);

	float **h_a_ptr_array;
	float **h_b_ptr_array;
	float **h_c_ptr_array;
	cudaMallocHost(&h_a_ptr_array, sizeof(float*) * batch_size);
	cudaMallocHost(&h_b_ptr_array, sizeof(float*) * batch_size);
	cudaMallocHost(&h_c_ptr_array, sizeof(float*) * batch_size);

	// Host memory for initializing
	float* init_matrix;
	cudaMallocHost(&init_matrix, sizeof(float) * m * n * k / (std::min(m, std::min(n, k))));
	for (unsigned i = 0; i < batch_size; i++) {
		// Allocate device memory and set
		float *d_a_ptr;
		float *d_b_ptr;
		float *d_c_ptr;
		cudaMalloc(&d_a_ptr, sizeof(float) * m * k);
		cudaMalloc(&d_b_ptr, sizeof(float) * k * n);
		cudaMalloc(&d_c_ptr, sizeof(float) * m * n);
		h_a_ptr_array[i] = d_a_ptr;
		h_b_ptr_array[i] = d_b_ptr;
		h_c_ptr_array[i] = d_c_ptr;

		// Initialize matrices
		// A
		for (unsigned j = 0; j < m * k; j++) init_matrix[j] = j / static_cast<float>(m * k);
		cudaMemcpy(d_a_ptr, init_matrix, sizeof(float) * m * k, cudaMemcpyDefault);
		// B
		for (unsigned j = 0; j < k * n; j++) init_matrix[j] = j / static_cast<float>(k * n);
		cudaMemcpy(d_b_ptr, init_matrix, sizeof(float) * k * n, cudaMemcpyDefault);
		// C
		for (unsigned j = 0; j < m * n; j++) init_matrix[j] = 0.f;
		cudaMemcpy(d_c_ptr, init_matrix, sizeof(float) * m * n, cudaMemcpyDefault);
	}
	cudaFreeHost(init_matrix);

	// Copy the pointer array to the device
	cudaMemcpy(d_a_ptr_array, h_a_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault);
	cudaMemcpy(d_b_ptr_array, h_b_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault);
	cudaMemcpy(d_c_ptr_array, h_c_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault);

	cudaDeviceSynchronize();
	bgemm<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(
			m, n, k,
			1.f,
			d_a_ptr_array, k,
			d_b_ptr_array, k,
			0.f,
			d_c_ptr_array, m,
			batch_size
			);
	cudaDeviceSynchronize();


	// evaluate the last batch matrix
	float* last_a_ptr;
	float* last_b_ptr;
	float* last_c_ptr;
	cudaMallocHost(&last_a_ptr, sizeof(float) * m * k);
	cudaMallocHost(&last_b_ptr, sizeof(float) * k * n);
	cudaMallocHost(&last_c_ptr, sizeof(float) * m * n);
	cudaMemcpy(last_a_ptr, h_a_ptr_array[batch_size - 1], sizeof(float) * m * k, cudaMemcpyDefault);
	cudaMemcpy(last_b_ptr, h_b_ptr_array[batch_size - 1], sizeof(float) * k * n, cudaMemcpyDefault);
	cudaMemcpy(last_c_ptr, h_c_ptr_array[batch_size - 1], sizeof(float) * m * n, cudaMemcpyDefault);
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
	cudaFree(last_a_ptr);
	cudaFree(last_b_ptr);
	cudaFree(last_c_ptr);

	cudaDeviceSynchronize();
	// evaluation of computing performance
	constexpr unsigned test_count = 1lu << 2;

	{
		cudaDeviceSynchronize();
		// evaluation of computing performance
		const auto start_clock = std::chrono::system_clock::now();
		for (unsigned c = 0; c < test_count; c++) {
			bgemm<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(
					m, n, k,
					1.f,
					d_a_ptr_array, k,
					d_b_ptr_array, k,
					0.f,
					d_c_ptr_array, m,
					batch_size
					);
		}
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;
		const auto complexity = 2lu * static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * static_cast<std::size_t>(batch_size);
		const auto performance = complexity / elapsed_time / (1lu << 40);

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
		const auto performance = complexity / elapsed_time / (1lu << 40);

		std::printf("%15s: %e s (cuBLAS)\n", "Time", elapsed_time);
		std::printf("%15s: %e TFlop/s (cuBLAS)\n", "Performance", performance);
		cublasDestroy(cublas_handle);
	}

	// Free
	for (unsigned i = 0; i < batch_size; i++) {
		cudaFree(h_a_ptr_array[i]);
		cudaFree(h_b_ptr_array[i]);
		cudaFree(h_c_ptr_array[i]);
	}
	cudaFree(d_a_ptr_array);
	cudaFree(d_b_ptr_array);
	cudaFree(d_c_ptr_array);
	cudaFreeHost(h_a_ptr_array);
	cudaFreeHost(h_b_ptr_array);
	cudaFreeHost(h_c_ptr_array);
}
} // noname napespace

int main() {
	test_batched_sgemm<128, 128, 16, 64, 32, 16, 256>(1024, 1024, 8192, 128);
}

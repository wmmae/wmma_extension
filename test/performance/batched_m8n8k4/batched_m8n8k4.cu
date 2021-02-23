#include <iostream>
#include <chrono>
#include <wmma_extension/wmma_extension.hpp>

constexpr unsigned warp_size = 32;
constexpr unsigned block_size = 256;

constexpr unsigned M = 8;
constexpr unsigned N = 8;
constexpr unsigned K = 4;

constexpr std::size_t num_matrices = 1lu << 20;
constexpr unsigned C = 1 << 8;

__global__ void batched_matmul_kernel(float* const c_ptr, const half* const a_ptr, const half* const b_ptr) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;

	if (matrix_id >= num_matrices) {
		return;
	}

	mtk::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::col_major> frag_a;
	mtk::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::col_major> frag_b;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> frag_c;

	mtk::wmma::load_matrix_sync(frag_a, a_ptr + matrix_id * M * K, M);
	mtk::wmma::load_matrix_sync(frag_b, b_ptr + matrix_id * N * K, K);
	mtk::wmma::fill_fragment(frag_c, 0.0f);

	mtk::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	mtk::wmma::store_matrix_sync(c_ptr + matrix_id * M * N, frag_c, M, nvcuda::wmma::mem_col_major);
}

int main() {
	half* da;
	half* db;
	float *dc;

	cudaMalloc(&da, sizeof(half) * M * K * num_matrices);
	cudaMalloc(&db, sizeof(half) * K * N * num_matrices);
	cudaMalloc(&dc, sizeof(float) * M * N * num_matrices);

	const auto start_clock = std::chrono::system_clock::now();
	for (unsigned c = 0; c < C; c++)
		batched_matmul_kernel<<<(warp_size * num_matrices + block_size - 1) / block_size, block_size>>>(dc, da, db);
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;

	std::printf("%15s : %e [s]\n", "elapsed time", elapsed_time);
	std::printf("%15s : %e [TFlop/s]\n", "performance", (2 * M * N * K * C * num_matrices) / elapsed_time / (1lu << 40));
	std::printf("%15s : %e [GiB/s]\n", "band width", static_cast<std::size_t>(M * N * sizeof(float) + 2 * (M * K + N * K) * sizeof(half)) * num_matrices * C / elapsed_time / (1lu << 30));

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

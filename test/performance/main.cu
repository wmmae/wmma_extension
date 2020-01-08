#include <iostream>
#include <chrono>
#include <type_traits>
#include <mma.h>
#include <wmma_extension.hpp>

//#define KERNEL_BREAKDOWN

template <class T>
__device__ void copy16x16(T* const dst_ptr, const T* const src_ptr, const unsigned unique_id) {
	constexpr unsigned dim = 16;
	constexpr unsigned warp_size = 32;
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

template <bool UseWMMAe, unsigned block_size>
__global__ void gemv16x16(float* const c_ptr, const half* const a_ptr, const half* const b_ptr) {
#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t1 = clock64();
#endif
	constexpr unsigned DIM = 16;
	constexpr unsigned warp_size = 32;

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned unique_id = threadIdx.x & 0x1f;
	const unsigned long matrix_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
	
	__shared__ half A_smem[DIM * DIM * block_size / warp_size];
	__shared__ float C_smem[DIM * DIM * block_size / warp_size];
	half* const A_smem_ptr = A_smem + warp_id * DIM * DIM;
	half* const B_smem_ptr = A_smem + warp_id * DIM * DIM;
	float* const C_smem_ptr = C_smem + warp_id * DIM * DIM;

#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t2 = clock64();
#endif
	copy16x16(A_smem_ptr, a_ptr + matrix_id * DIM * DIM, unique_id);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, DIM, DIM, DIM, half, nvcuda::wmma::col_major> A_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, DIM, DIM, DIM, half, nvcuda::wmma::col_major> B_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, DIM, DIM, DIM, float> C_frag;

	nvcuda::wmma::fill_fragment(C_frag, 0.0f);
	nvcuda::wmma::load_matrix_sync(A_frag, A_smem_ptr, DIM);
#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t3 = clock64();
#endif

	if (!UseWMMAe) {
		fill16x16(B_smem_ptr, __float2half(0.0f), unique_id);
	}
	copy16(B_smem_ptr, b_ptr + matrix_id * DIM, unique_id);
#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t4 = clock64();
#endif

	if (UseWMMAe) {
		mtk::wmma::load_vector_sync(B_frag, B_smem_ptr);
	} else {
		nvcuda::wmma::load_matrix_sync(B_frag, B_smem_ptr, DIM);
	}
#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t5 = clock64();
#endif

	nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t6 = clock64();
#endif

	nvcuda::wmma::store_matrix_sync(C_smem_ptr, C_frag, DIM, nvcuda::wmma::mem_col_major);
	copy16x16(c_ptr + matrix_id * DIM * DIM, C_smem_ptr, unique_id);
#ifdef KERNEL_BREAKDOWN
	__syncthreads();
	const auto t7 = clock64();
	if (unique_id == 0)
		printf("%lu,%lu,%lu,%lu,%lu,%lu\n",
				(t2 - t1),
				(t3 - t2),
				(t4 - t3),
				(t5 - t4),
				(t6 - t5),
				(t7 - t6)
				);
#endif
}

template <bool UseWMMAe>
void test_gemv() {
	constexpr unsigned DIM = 16;
	constexpr unsigned warp_size = 32;
	constexpr std::size_t num_gemv = 1lu << 15;
	constexpr std::size_t C = 1lu << 4;
	constexpr std::size_t block_size = 256;
	constexpr std::size_t grid_size = (num_gemv * warp_size) / block_size;

	half *dA, *dB;
	float *dC;
	cudaMalloc(&dA, sizeof(half) * DIM * DIM * num_gemv);
	cudaMalloc(&dB, sizeof(half) * DIM * num_gemv);
	cudaMalloc(&dC, sizeof(float) * DIM * DIM * num_gemv);

#ifdef KERNEL_BREAKDOWN
	std::printf("init_p,init_ca_frag,init_b_smem,init_b_frag,mma,store_c\n");
#endif

	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t c = 0; c < C; c++) {
		gemv16x16<UseWMMAe, block_size><<<grid_size, block_size>>>(
				dC,
				dA,
				dB
				);
	}
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / 1.e6 / C;
	std::printf("%u,%e,%e\n",
			(UseWMMAe ? 1u : 0u),
			elapsed_time,
			(2 * DIM * DIM / elapsed_time / (1lu<<40)));
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}


int main() {
	test_gemv<false>();
	test_gemv<true>();
}

#include <iostream>
#include <chrono>
#include <type_traits>
#include <mma.h>
#include <wmma_extension.hpp>

constexpr std::size_t block_size = 256;
constexpr unsigned warp_size = 32;

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
__global__ void direct_product16x16(float* const c_ptr, const half* const a_ptr, const half* const b_ptr, unsigned dim);

template <>
__global__ void direct_product16x16<true>(float* const c_ptr, const half* const a_ptr, const half* const b_ptr, unsigned dim) {
	constexpr unsigned FDIM = 16;
	__shared__ half A_smem[warp_size];
	__shared__ half B_smem[block_size];
	__shared__ float C_smem[warp_size * block_size];

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned unique_id = threadIdx.x & 0x1;
	const unsigned a_start = blockIdx.x * warp_size;

	float* const C_smem_ptr = C_smem + warp_size * FDIM * warp_id;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> A_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> B_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> C_frag[2];

	// load A
	if (warp_id == 0) {
		A_smem[unique_id] = a_ptr[a_start + unique_id];
	}
	__syncthreads();
	mtk::wmma::load_vector_sync(A_frag[0], A_smem);
	mtk::wmma::load_vector_sync(A_frag[1], A_smem + FDIM);


	for (unsigned b_start = 0; b_start < dim; b_start += block_size) {
		nvcuda::wmma::fill_fragment(C_frag[0], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[1], __float2half(0.0f));
		// load B
		B_smem[unique_id] = b_ptr[unique_id];
		mtk::wmma::load_vector_sync(B_frag, B_smem);

		nvcuda::wmma::mma_sync(C_frag[0], A_frag[0], B_frag, C_frag[0]);
		nvcuda::wmma::mma_sync(C_frag[1], A_frag[1], B_frag, C_frag[1]);

		nvcuda::wmma::store_matrix_sync(C_smem_ptr, C_frag[0], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM, C_frag[1], warp_size, nvcuda::wmma::mem_col_major);

		const unsigned h = block_size / warp_size;
		const unsigned c_start = warp_id * h;
		for (unsigned i = 0; i < h; i++) {
			c_ptr[dim * (b_start + c_start + i) + unique_id] = C_smem_ptr[i * warp_size + unique_id];
		}

	}
}

template <>
__global__ void direct_product16x16<false>(float* const c_ptr, const half* const a_ptr, const half* const b_ptr, unsigned dim) {
	constexpr unsigned FDIM = 16;
	__shared__ half B_smem[block_size * warp_size];
	half* const A_smem = B_smem;
	__shared__ float C_smem[warp_size * block_size];

	const unsigned warp_id = threadIdx.x >> 5;
	const unsigned unique_id = threadIdx.x & 0x1;
	const unsigned a_start = blockIdx.x * warp_size;

	float* const C_smem_ptr = C_smem + warp_size * FDIM * warp_id;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half, nvcuda::wmma::col_major> A_frag[2];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half, nvcuda::wmma::row_major> B_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float> C_frag[2];

	// load A
	if (warp_id == 0) {
		A_smem[unique_id] = a_ptr[a_start + unique_id];
		for (unsigned i = 1; i < warp_size; i++) {
			A_smem[unique_id + i * warp_size] = __float2half(0.0f);
		}
	}
	__syncthreads();
	mtk::wmma::load_vector_sync(A_frag[0], A_smem);
	mtk::wmma::load_vector_sync(A_frag[1], A_smem + FDIM);
	// init B
	const unsigned h = block_size / warp_size;
	for (unsigned i = 0; i < h; i++) {
		B_smem[i * warp_size + unique_id] = __float2half(0.0f);
	}
	__syncthreads();


	for (unsigned b_start = 0; b_start < dim; b_start += block_size) {
		nvcuda::wmma::fill_fragment(C_frag[0], __float2half(0.0f));
		nvcuda::wmma::fill_fragment(C_frag[1], __float2half(0.0f));
		// load B
		B_smem[unique_id] = b_ptr[unique_id];
		nvcuda::wmma::load_matrix_sync(B_frag, B_smem, warp_size);

		nvcuda::wmma::mma_sync(C_frag[0], A_frag[0], B_frag, C_frag[0]);
		nvcuda::wmma::mma_sync(C_frag[1], A_frag[1], B_frag, C_frag[1]);

		nvcuda::wmma::store_matrix_sync(C_smem_ptr, C_frag[0], warp_size, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::store_matrix_sync(C_smem_ptr + FDIM, C_frag[1], warp_size, nvcuda::wmma::mem_col_major);

		const unsigned c_start = warp_id * h;
		for (unsigned i = 0; i < h; i++) {
			c_ptr[dim * (b_start + c_start + i) + unique_id] = C_smem_ptr[i * warp_size + unique_id];
		}
	}
}

template <bool UseWMMAe>
void test_direct_product() {
	constexpr unsigned DIM = 1lu << 14;
	constexpr std::size_t C = 1lu << 10;
	constexpr std::size_t grid_size = DIM / warp_size;

	half *dA, *dB;
	float *dC;
	cudaMalloc(&dA, sizeof(half) * DIM);
	cudaMalloc(&dB, sizeof(half) * DIM);
	cudaMalloc(&dC, sizeof(float) * DIM * DIM);

#ifdef KERNEL_BREAKDOWN
	std::printf("init_p,init_ca_frag,init_b_smem,init_b_frag,mma,store_c\n");
#endif

	const auto start_clock = std::chrono::system_clock::now();
	for (std::size_t c = 0; c < C; c++) {
		direct_product16x16<UseWMMAe><<<grid_size, block_size>>>(
				dC,
				dA,
				dA,
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
	std::printf("%u,%e,%e\n",
			(UseWMMAe ? 1u : 0u),
			elapsed_time,
			(2 * DIM * DIM / elapsed_time / (1lu<<40)));
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}


int main() {
	test_direct_product<false>();
	test_direct_product<true>();
}

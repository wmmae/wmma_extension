#include <stdio.h>
#include <wmma_extension.hpp>

constexpr std::size_t N = 16;
constexpr unsigned warp_size = 32;

__global__ void wlv_matrix_a_test_kernel(const float* const ptr) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;
	//nvcuda::wmma::load_matrix_sync(frag_a, ptr, N);
	mtk::wmma::load_vector_sync(frag_a, ptr);

	for(unsigned i = 0; i < warp_size; i++) {
		if(i == threadIdx.x) {
			for(unsigned j = 0; j < frag_a.num_elements; j++) {
				printf("%03d ", (int)__half2float(frag_a.x[j]));
			}
			printf("\n");
		}
		__syncthreads();
	}
}

__global__ void wlv_matrix_b_test_kernel(const float* const ptr) {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::col_major> frag_b;
	//nvcuda::wmma::load_matrix_sync(frag_a, ptr, N);
	mtk::wmma::load_vector_sync(frag_b, ptr);

	for(unsigned i = 0; i < warp_size; i++) {
		if(i == threadIdx.x) {
			for(unsigned j = 0; j < frag_b.num_elements; j++) {
				printf("%03d ", (int)__half2float(frag_b.x[j]));
			}
			printf("\n");
		}
		__syncthreads();
	}
}

__global__ void make_identity_test_kernel() {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, N, N, half> frag_c;
	//nvcuda::wmma::load_matrix_sync(frag_a, ptr, N);
	mtk::wmma::make_identity_matrix_sm70(frag_c);

	for(unsigned i = 0; i < warp_size; i++) {
		if(i == threadIdx.x) {
			for(unsigned j = 0; j < frag_c.num_elements; j++) {
				printf("%03d ", (int)__half2float(frag_c.x[j]));
			}
			printf("\n");
		}
		__syncthreads();
	}
}

__global__ void syrk_test_kernel(const float* const ptr, half* const result_ptr) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, N, N, N, half> frag_c;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::row_major> frag_b;
	nvcuda::wmma::fill_fragment(frag_c, __float2half(0.0f));
	mtk::wmma::load_vector_sync_sm70(frag_a, ptr);
	mtk::wmma::load_vector_sync_sm70(frag_b, ptr);
	mtk::wmma::make_identity_matrix_sm70(frag_c);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(result_ptr, frag_c, N, nvcuda::wmma::mem_col_major);
}

int main() {
	float *matrix;
	cudaMallocHost((void**)&matrix, sizeof(float) * N * N);
	half *result_matrix;
	cudaMallocHost((void**)&result_matrix, sizeof(half) * N * N);

	for(std::size_t i = 0; i < N * N; i++) {
		matrix[i] = static_cast<float>(i + 1);
	}

	printf("matrix_a test\n");
	wlv_matrix_a_test_kernel<<<1, warp_size>>>(matrix);
	cudaDeviceSynchronize();
	printf("matrix_b test\n");
	wlv_matrix_b_test_kernel<<<1, warp_size>>>(matrix);
	cudaDeviceSynchronize();
	printf("matrix_c test\n");
	make_identity_test_kernel<<<1, warp_size>>>();
	cudaDeviceSynchronize();
	printf("syrk test\n");
	syrk_test_kernel<<<1, warp_size>>>(matrix, result_matrix);
	for(std::size_t i = 0; i < N; i++) {
		for(std::size_t j = 0; j < N; j++) {
			printf("%03.f ", __half2float(result_matrix[j * N + i]));
		}
		printf("\n");
	}
}

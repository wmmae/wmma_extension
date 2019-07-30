#include <stdio.h>
#include <wmma_load_vector.hpp>

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

int main() {
	float *matrix;
	cudaMallocHost((void**)&matrix, sizeof(float) * N * N);

	for(std::size_t i = 0; i < N * N; i++) {
		matrix[i] = static_cast<float>(i + 1);
	}

	printf("matrix_a test\n");
	wlv_matrix_a_test_kernel<<<1, warp_size>>>(matrix);
	cudaDeviceSynchronize();
	printf("matrix_b test\n");
	wlv_matrix_b_test_kernel<<<1, warp_size>>>(matrix);
	cudaDeviceSynchronize();
}

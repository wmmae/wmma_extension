#include <iostream>
#include <wmma_extension.hpp>

template <class T>
__global__ void test_load_vector_kernel(
		T* const dst,
		const T* const src,
		const nvcuda::wmma::layout_t layout
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;
	nvcuda::wmma::load_matrix_sync(frag_c, src, 16, layout);
	mtk::wmma::store_vector_sync(dst, frag_c, layout);
}

void test(const nvcuda::wmma::layout_t layout) {
	std::printf("-- store_vector test --\n");
	if (layout == nvcuda::wmma::mem_col_major) {
		std::printf("layout : col_major\n");
	} else {
		std::printf("layout : row_major\n");
	}
	float* src_mem;
	float* dst_mem;

	cudaMallocHost(&src_mem, 16 * 16 * sizeof(float));
	cudaMallocHost(&dst_mem, 16 * sizeof(float));

	for (std::size_t i = 0; i < 16 * 16; i++) {
		src_mem[i] = i;
	}

	test_load_vector_kernel<<<1, 32>>>(dst_mem, src_mem, layout);
	cudaDeviceSynchronize();

	for (std::size_t i = 0; i < 16; i++) {
		std::printf("%3.1f ", dst_mem[i]);
	}
	std::printf("\n");
}

int main() {
	test(nvcuda::wmma::mem_col_major);
	test(nvcuda::wmma::mem_row_major);
}

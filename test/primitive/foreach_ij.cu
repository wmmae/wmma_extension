#include <iostream>
#include <random>
#include <type_traits>
#include <math.h>
#include <wmma_extension/wmma_extension.hpp>
#include "common.hpp"

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

__device__ float myabs(const float a) {
	if (a > 0) {
		return a;
	} else {
		return -a;
	}
}

template <class Use, int M, int N, int K, class Type, class Layout, unsigned MATRIX_DIM>
__global__ void test_kernel(float* const diff, const float* const src, const unsigned ld) {
	using storage_t = typename mtk::wmma::detail::common::storage_t<Type>::type;

	__shared__ storage_t smem[MATRIX_DIM * MATRIX_DIM];
	for (unsigned i = 0; i < MATRIX_DIM * MATRIX_DIM; i += blockDim.x) {
		smem[i + threadIdx.x] = src[i + threadIdx.x];
	}

	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> frag_nvcuda;
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> frag_mtk;

	nvcuda::wmma::load_matrix_sync(frag_nvcuda, smem, ld);

	mtk::wmma::foreach_ij<decltype(frag_mtk)>(
			[&](const unsigned* frag_index_list, const unsigned num_indeces, const unsigned i, const unsigned j) {
				unsigned mem_index;
				if (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
					mem_index = i + j * ld;
				} else {
					mem_index = i * ld + j;
				}
				for (unsigned f = 0; f < num_indeces; f++) {
					frag_mtk.x[frag_index_list[f]] = smem[mem_index];
				}
			}
			);

	float max_diff = 0.f;
	for (unsigned i = 0; i < frag_mtk.num_elements; i++) {
		max_diff = max(max_diff, myabs(frag_mtk.x[i] - frag_nvcuda.x[i]));
	}
	diff[threadIdx.x] = max_diff;
}

template <class Use, int M, int N, int K, class Type, nvcuda::wmma::layout_t layout, unsigned MATRIX_DIM>
__global__ void test_kernel_acc(float* const diff, const float* const src, const unsigned ld) {
	using storage_t = typename mtk::wmma::detail::common::storage_t<Type>::type;

	__shared__ storage_t smem[MATRIX_DIM * MATRIX_DIM];
	for (unsigned i = 0; i < MATRIX_DIM * MATRIX_DIM; i += blockDim.x) {
		smem[i + threadIdx.x] = src[i + threadIdx.x];
	}

	nvcuda::wmma::fragment<Use, M, N, K, Type, void> frag_nvcuda;
	nvcuda::wmma::fragment<Use, M, N, K, Type, void> frag_mtk;

	nvcuda::wmma::load_matrix_sync(frag_nvcuda, smem, ld, layout);

	mtk::wmma::foreach_ij<decltype(frag_mtk)>(
			layout,
			[&](const unsigned* frag_index_list, const unsigned num_indeces, const unsigned i, const unsigned j) {
				unsigned mem_index;
				if (layout == nvcuda::wmma::mem_col_major) {
					mem_index = i + j * ld;
				} else {
					mem_index = i * ld + j;
				}
				for (unsigned f = 0; f < num_indeces; f++) {
					frag_mtk.x[frag_index_list[f]] = smem[mem_index];
				}
			}
			);

	float max_diff = 0.f;
	for (unsigned i = 0; i < frag_mtk.num_elements; i++) {
		max_diff = max(max_diff, myabs(frag_mtk.x[i] - frag_nvcuda.x[i]));
	}
	diff[threadIdx.x] = max_diff;
}

template <class Use, int M, int N, int K, class Type, class Layout>
void test() {
	constexpr unsigned MATRIX_DIM = 32;
	constexpr unsigned warp_size = 32;
	float* src_matrix;
	float* diff;
	cudaMallocHost(&src_matrix, sizeof(float) * MATRIX_DIM * MATRIX_DIM);
	cudaMallocHost(&diff, sizeof(float) * warp_size);

	for (unsigned i = 0; i < MATRIX_DIM * MATRIX_DIM; i++) {
		src_matrix[i] = static_cast<float>(i) / (MATRIX_DIM);
	}

	test_kernel<Use, M, N, K, Type, Layout, MATRIX_DIM><<<1, warp_size>>>(diff, src_matrix, MATRIX_DIM);
	cudaDeviceSynchronize();

	bool passed = true;
	for (unsigned i = 0; i < warp_size; i++) {
		if (diff[i] > (1.f / MATRIX_DIM / 2)) {
			passed = false;
		}
	}

	std::printf("%s{SM=%2d,Use=%15s,M=%2d,N=%2d,K=%2d,Type=%5s,Layout=%8s}:",
			__FILE__,
			TEST_ARCH,
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<Type>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str()
			);
	if (passed) {
		std::printf("PASSED");
	} else {
		std::printf("FAILED");
	}
	std::printf("\n");

	cudaFreeHost(diff);
	cudaFreeHost(src_matrix);
}

template <class Use, int M, int N, int K, class Type, class Layout>
void test_acc() {
	constexpr unsigned MATRIX_DIM = 32;
	constexpr unsigned warp_size = 32;
	float* src_matrix;
	float* diff;
	cudaMallocHost(&src_matrix, sizeof(float) * MATRIX_DIM * MATRIX_DIM);
	cudaMallocHost(&diff, sizeof(float) * warp_size);

	for (unsigned i = 0; i < MATRIX_DIM * MATRIX_DIM; i++) {
		src_matrix[i] = static_cast<float>(i) / (MATRIX_DIM);
	}

	if (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
		test_kernel_acc<Use, M, N, K, Type, nvcuda::wmma::mem_col_major, MATRIX_DIM><<<1, warp_size>>>(diff, src_matrix, MATRIX_DIM);
	} else {
		test_kernel_acc<Use, M, N, K, Type, nvcuda::wmma::mem_row_major, MATRIX_DIM><<<1, warp_size>>>(diff, src_matrix, MATRIX_DIM);
	}
	cudaDeviceSynchronize();

	bool passed = true;
	for (unsigned i = 0; i < warp_size; i++) {
		if (diff[i] > (1.f / MATRIX_DIM / 2)) {
			passed = false;
		}
	}

	std::printf("%s{SM=%2d,Use=%15s,M=%2d,N=%2d,K=%2d,Type=%5s,Layout=%8s}:",
			__FILE__,
			TEST_ARCH,
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<Type>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str()
			);
	if (passed) {
		std::printf("PASSED");
	} else {
		std::printf("FAILED");
	}
	std::printf("\n");

	cudaFreeHost(diff);
	cudaFreeHost(src_matrix);
}

int main() {
	test<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>();
	test_acc<nvcuda::wmma::accumulator, 16, 16, 16, half , nvcuda::wmma::col_major>();
	test_acc<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::col_major>();
	test_acc<nvcuda::wmma::accumulator, 16, 16, 16, half , nvcuda::wmma::row_major>();
	test_acc<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::row_major>();
#ifdef TEST_TF32
	test<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test_acc<nvcuda::wmma::accumulator, 16, 16, 8, float, nvcuda::wmma::col_major>();
	test_acc<nvcuda::wmma::accumulator, 16, 16, 8, float, nvcuda::wmma::row_major>();
#endif
}

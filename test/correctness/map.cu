#include <iostream>
#include <wmma_extension/wmma_extension.hpp>
#include "common.hpp"

//#define TEST_TF32

constexpr unsigned warp_size = 32;

__device__ inline float my_abs(float a) {
	return (a < 0) ? (-a) : a;
}

template <class Use, int M, int N, int K, class T, class Layout>
__global__ void map_test(float* const error_ptr) {
	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	constexpr unsigned mat_m = mtk::wmma::detail::common::get_M<Use, M, N, K>::value;
	constexpr unsigned mat_n = mtk::wmma::detail::common::get_N<Use, M, N, K>::value;
	__shared__ storage_t smem[mat_m * mat_n];

	for (unsigned i = 0; i < mat_m * mat_n; i += warp_size) {
		const unsigned index = i + threadIdx.x;
		const auto m = std::is_same<Layout, nvcuda::wmma::row_major>::value ? (index / mat_n) : (index % mat_m);
		const auto n = std::is_same<Layout, nvcuda::wmma::row_major>::value ? (index % mat_n) : (index / mat_m);
		smem[index] = mtk::wmma::detail::common::cast<storage_t>(m + n * mat_m);
	}

	nvcuda::wmma::fragment<Use, M, N, K, T, Layout> frag_ref;
	nvcuda::wmma::load_matrix_sync(frag_ref, smem, std::is_same<Layout, nvcuda::wmma::row_major>::value ? mat_n : mat_m);

	nvcuda::wmma::fragment<Use, M, N, K, T, Layout> frag_map;
	for (unsigned i = 0; i < mat_m; i++) {
		for (unsigned j = 0; j < mat_n; j++) {
			unsigned tid_list[2];
			unsigned fid_list[2];
			unsigned list_size;
			mtk::wmma::map<decltype(frag_map)>(tid_list, fid_list, list_size, i, j);

			for (unsigned k = 0; k < list_size; k++) {
				if (threadIdx.x == tid_list[k]) {
					frag_map.x[fid_list[k]] = mtk::wmma::detail::common::cast<storage_t>(i + j * mat_m);
				}
				__syncwarp();
			}
		}
	}
	float error = 0.f;
	for (unsigned i = 0; i < frag_map.num_elements; i++) {
		error += my_abs(frag_map.x[i] - frag_ref.x[i]);
	}

	atomicAdd(error_ptr, error);
}

template <class Use, int M, int N, int K, class T>
__global__ void map_test(float* const error_ptr, const nvcuda::wmma::layout_t layout) {
	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	constexpr unsigned mat_m = mtk::wmma::detail::common::get_M<Use, M, N, K>::value;
	constexpr unsigned mat_n = mtk::wmma::detail::common::get_N<Use, M, N, K>::value;
	__shared__ storage_t smem[mat_m * mat_n];

	for (unsigned i = 0; i < mat_m * mat_n; i += warp_size) {
		const unsigned index = i + threadIdx.x;
		const float v = (layout == nvcuda::wmma::mem_row_major) ? ((index / mat_n) + (index % mat_n) * mat_m) : index;
		smem[index] = mtk::wmma::detail::common::cast<storage_t>(v);
	}
	for (unsigned i = 0; i < mat_m * mat_n; i += warp_size) {
		const unsigned index = i + threadIdx.x;
		const auto m = (layout == nvcuda::wmma::mem_row_major) ? (index / mat_n) : (index % mat_m);
		const auto n = (layout == nvcuda::wmma::mem_row_major) ? (index % mat_n) : (index / mat_m);
		smem[index] = mtk::wmma::detail::common::cast<storage_t>(m + n * mat_m);
	}

	nvcuda::wmma::fragment<Use, M, N, K, T> frag_ref;
	nvcuda::wmma::load_matrix_sync(frag_ref, smem, (layout == nvcuda::wmma::mem_row_major) ? mat_n : mat_m, layout);

	nvcuda::wmma::fragment<Use, M, N, K, T> frag_map;
	for (unsigned i = 0; i < mat_m; i++) {
		for (unsigned j = 0; j < mat_n; j++) {
			unsigned tid_list[2];
			unsigned fid_list[2];
			unsigned list_size;
			mtk::wmma::map<decltype(frag_map)>(tid_list, fid_list, list_size, i, j);

			for (unsigned k = 0; k < list_size; k++) {
				if (threadIdx.x == tid_list[k]) {
					frag_map.x[fid_list[k]] = mtk::wmma::detail::common::cast<storage_t>(i + j * mat_m);
				}
				__syncwarp();
			}
		}
	}
	float error = 0.f;
	for (unsigned i = 0; i < frag_map.num_elements; i++) {
		error += my_abs(frag_map.x[i] - frag_ref.x[i]);
	}

	atomicAdd(error_ptr, error);
}

template <class Use, int M, int N, int K, class T, class Layout>
void test() {
	float* error;
	cudaMallocHost(&error, sizeof(float));
	map_test<Use, M, N, K, T, Layout><<<1, warp_size>>>(error);
	cudaDeviceSynchronize();

	std::printf("%s<%12s,%2d,%2d,%2d,%7s,%10s>:Error=%e [",
			__FILE__,
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<T>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str(),
			*error
			);
	if (*error < 1) {
		std::printf("PASSED");
	} else {
		std::printf("FAILED");
	}
	std::printf("]\n");
	cudaFreeHost(error);
}

template <class Use, int M, int N, int K, class T>
void test(const nvcuda::wmma::layout_t layout) {
	float* error;
	cudaMallocHost(&error, sizeof(float));
	map_test<Use, M, N, K, T><<<1, warp_size>>>(error, layout);
	cudaDeviceSynchronize();

	std::printf("%s<%12s,%2d,%2d,%2d,%7s,%10s>:Error=%e [",
			__FILE__,
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<T>().c_str(),
			(layout == nvcuda::wmma::mem_col_major) ? mtk::test_utils::get_string<nvcuda::wmma::col_major>().c_str() : mtk::test_utils::get_string<nvcuda::wmma::row_major>().c_str(),
			*error
			);
	if (*error < 1) {
		std::printf("PASSED");
	} else {
		std::printf("FAILED");
	}
	std::printf("]\n");
	cudaFreeHost(error);
}

int main() {
	test<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 16, half>(nvcuda::wmma::mem_col_major);
	test<nvcuda::wmma::accumulator, 16, 16, 16, float>(nvcuda::wmma::mem_col_major);
	test<nvcuda::wmma::accumulator, 16, 16, 16, half>(nvcuda::wmma::mem_row_major);
	test<nvcuda::wmma::accumulator, 16, 16, 16, float>(nvcuda::wmma::mem_row_major);

#ifdef TEST_TF32
	test<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 8, float>(nvcuda::wmma::mem_col_major);
	test<nvcuda::wmma::accumulator, 16, 16, 8, float>(nvcuda::wmma::mem_row_major);
#endif
}

#include <iostream>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

//#define TEST_TF32

#ifndef TEST_TF32
constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr std::size_t K = 16;
#else
constexpr std::size_t M = 16;
constexpr std::size_t N = 16;
constexpr std::size_t K = 8;
#endif

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

__global__ void test_foreach_v_acc_kernel(
		float* const dst,
		const float* const src,
		const nvcuda::wmma::layout_t layout
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> frag_c;
	nvcuda::wmma::load_matrix_sync(frag_c, src, M, layout);
	mtk::wmma::foreach_v<decltype(frag_c)>(layout, [&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
				for (unsigned i = 0; i < fragment_index_count; i++) {
					dst[mem_index] = frag_c.x[frag_index_list[i]];
				}
			});
}

void test(const nvcuda::wmma::layout_t layout) {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch   : %d\n", TEST_ARCH);
	if (layout == nvcuda::wmma::mem_col_major) {
		std::printf("layout : col_major\n");
	} else {
		std::printf("layout : row_major\n");
	}
	std::printf("size   : %lu, %lu, %lu\n", M, N, K);
	float* src_mem;
	float* dst_mem;

	cudaMallocHost(&src_mem, M * N * sizeof(float));
	cudaMallocHost(&dst_mem, M * sizeof(float));

	for (std::size_t i = 0; i < M * N; i++) {
		src_mem[i] = static_cast<float>(i);
	}

	cudaDeviceSynchronize();
	test_foreach_v_acc_kernel<<<1, 32>>>(dst_mem, src_mem, layout);
	cudaDeviceSynchronize();

	float error = 0.f;
	for (std::size_t i = 0; i < M; i++) {
		error = std::max(std::abs(dst_mem[i] - src_mem[i]), error);
	}
	std::printf("error = %e\n", error);
}

int main() {
	test(nvcuda::wmma::mem_row_major);
	test(nvcuda::wmma::mem_col_major);
	test(nvcuda::wmma::mem_row_major);
	test(nvcuda::wmma::mem_col_major);
}

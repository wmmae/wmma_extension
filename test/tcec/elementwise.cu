#include <iostream>
#include <wmma_extension/tcec/tcec.hpp>
#include "utils.hpp"

constexpr unsigned warp_size = 32;

template <unsigned N, class T, class Policy>
__global__ void test_elementwise_kernel(float* const ptr) {
	__shared__ float smem[N * N];

	mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, N, N, N, T, void, Policy> frag;
	mtk::wmma::tcec::fill_fragment(frag, 0.0f);

	for (unsigned i = 0; i < frag.num_elements; i++) {
		frag.x(i) = threadIdx.x * 100 + i;
	}

	mtk::wmma::tcec::store_matrix_sync(smem, frag, N, nvcuda::wmma::mem_col_major);

	for (unsigned i = 0; i < N * N; i += warp_size) {
		const auto index = i + threadIdx.x;
		ptr[index] = smem[index];
	}
}

template <unsigned N, class T, class Policy>
void test_elementwise() {
	std::printf("[%s, N = %u, T = %s, Policy = <%7s,%9s,%2u,%2u,%2u>]\n",
			__func__,
			N,
			mtk::test_utils::to_string<T>().c_str(),
			mtk::test_utils::to_string<typename Policy::op>().c_str(),
			std::is_same<typename Policy::error_correction, mtk::wmma::tcec::with_ec>::value ? "{w/ ec}" : "{w/o ec}",
			Policy::m,
			Policy::n,
			Policy::k
			);
	float* hC;
	cudaMallocHost(&hC, N * N * sizeof(float));

	test_elementwise_kernel<N, T, Policy><<<1, warp_size>>>(hC);
	cudaDeviceSynchronize();

	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			std::printf("%e ", hC[i + j * N]);
		}
		std::printf("\n");
	}
}

int main() {
	test_elementwise<32, half                         , typename mtk::wmma::tcec::detail::default_policy<half                         , mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	test_elementwise<32, half                         , typename mtk::wmma::tcec::detail::default_policy<half                         , mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	test_elementwise<32, half                         , typename mtk::wmma::tcec::detail::default_policy<half                         , mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_mma >::type>();
	test_elementwise<32, half                         , typename mtk::wmma::tcec::detail::default_policy<half                         , mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma >::type>();
	test_elementwise<32, float                        , typename mtk::wmma::tcec::detail::default_policy<float                        , mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();

#ifdef TEST_SIMT
	test_elementwise<32, float                        , typename mtk::wmma::tcec::detail::default_policy<float                        , mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
#endif
#ifdef TEST_TF32
	test_elementwise<32, nvcuda::wmma::precision::tf32, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	test_elementwise<32, nvcuda::wmma::precision::tf32, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
#endif
}

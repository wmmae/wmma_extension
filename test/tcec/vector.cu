#include <iostream>
#include <type_traits>
#include <wmma_extension/tcec/tcec.hpp>
#include "utils.hpp"

__device__ half abs(const half a) {
	if (__half2float(a) < 0) {
		return -a;
	}
	return a;
}

/// Load

template <class Use, int m, int n, int k, class T, class Layout, class Policy>
__global__ void load_vector_ab_test_kernel(
		const float* const cor_ptr,
		const float* const src_ptr
		) {
	mtk::wmma::tcec::fragment<Use, m, n, k, T, Layout, Policy> frag, frag_c;
	mtk::wmma::tcec::fill_fragment(frag, 0.0f);

	mtk::wmma::tcec::load_vector(frag, src_ptr);

	constexpr unsigned mem_m = mtk::wmma::tcec::detail::select_value<Use, m, k, m>::value;
	mtk::wmma::tcec::load_matrix_sync(frag_c, cor_ptr, mem_m);

	float max_error = 0;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		max_error = max(max_error, abs(frag.x(i) - frag_c.x(i)));
	}
	for (unsigned i = 0; i < mtk::test_utils::warp_size; i++) {
		__syncthreads();
		if (i == threadIdx.x) printf("[%u] %e\n", i, max_error);
	}
}

template <int m, int n, int k, class T, class Policy>
__global__ void load_vector_acc_test_kernel(
		const float* const cor_ptr,
		const float* const src_ptr,
		const nvcuda::wmma::layout_t layout
		) {
	mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, m, n, k, T, void, Policy> frag, frag_c;
	mtk::wmma::tcec::fill_fragment(frag, 0.0f);

	mtk::wmma::tcec::load_vector(frag, src_ptr, layout);

	constexpr unsigned mem_m = m;
	mtk::wmma::tcec::load_matrix_sync(frag_c, cor_ptr, mem_m, layout);

	float max_error = 0;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		max_error = max(max_error, abs(frag.x(i) - frag_c.x(i)));
	}
	for (unsigned i = 0; i < mtk::test_utils::warp_size; i++) {
		__syncthreads();
		if (i == threadIdx.x) printf("[%u] %e\n", i, max_error);
	}
}

template <class Use, int m, int n, int k, class T, class Layout, class Policy>
struct test_switch {
	void operator()(float* const mat_mem, float* const vec_mem) {
		load_vector_ab_test_kernel<Use, m, n, k, T, Layout, Policy><<<1, mtk::test_utils::warp_size>>>(mat_mem, vec_mem);
	}
};
template <int m, int n, int k, class T, class Layout, class Policy>
struct test_switch<nvcuda::wmma::accumulator, m, n, k, T, Layout, Policy> {
	void operator()(float* const mat_mem, float* const vec_mem) {
		const auto layout = (std::is_same<nvcuda::wmma::col_major, Layout>::value) ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major;
		load_vector_acc_test_kernel<m, n, k, T, Policy><<<1, mtk::test_utils::warp_size>>>(mat_mem, vec_mem, layout);
	}
};

template <class Use, int m, int n, int k, class T, class Layout, class Policy>
void load_vector_test() {
	std::printf("!-- %s\n", __func__);
	std::printf("Use    : %s\n", mtk::test_utils::to_string<Use>().c_str());
	std::printf("Layout : %s\n", mtk::test_utils::to_string<Layout>().c_str());
	std::printf("Type   : %s\n", mtk::test_utils::to_string<T>().c_str());
	std::printf("Size   : %u, %u, %u\n", m, n, k);
	std::printf("Policy : Policy<%7s,%9s,%2u,%2u,%2u>\n",
			mtk::test_utils::to_string<typename Policy::op>().c_str(),
			std::is_same<typename Policy::error_correction, mtk::wmma::tcec::with_ec>::value ? "{w/ ec}" : "{w/o ec}",
			Policy::m,
			Policy::n,
			Policy::k);
	constexpr unsigned mem_m = mtk::wmma::tcec::detail::select_value<Use, m, k, m>::value;
	constexpr unsigned mem_n = mtk::wmma::tcec::detail::select_value<Use, k, n, n>::value;

	constexpr auto vec_len = std::is_same<Layout, nvcuda::wmma::col_major>::value ? mem_m : mem_n;

	float* vec_mem;
	float* mat_mem;

	cudaMallocHost(&mat_mem, sizeof(float) * mem_m * mem_n);
	cudaMallocHost(&vec_mem, sizeof(float) * vec_len);

	for (unsigned i = 0; i < vec_len; i++) {
		vec_mem[i] = i;
	}

	for (unsigned i = 0; i < mem_m * mem_n; i++) {
		mat_mem[i] = 0.f;
	}

	for (unsigned i = 0; i < vec_len; i++) {
		mat_mem[i] = vec_mem[i];
	}

	test_switch<Use, m, n, k, T, Layout, Policy> test;
	test(mat_mem, vec_mem);

	cudaDeviceSynchronize();

	cudaFree(vec_mem);
	cudaFree(mat_mem);
}

/// Store

template <int m, int n, int k, class T, class Policy>
__global__ void store_vector_acc_test_kernel(
		float* const dst_ptr,
		const float* const src_ptr,
		const nvcuda::wmma::layout_t layout
		) {
	mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, m, n, k, T, void, Policy> frag;

	constexpr unsigned mem_m = m;
	mtk::wmma::tcec::load_matrix_sync(frag, src_ptr, mem_m, layout);

	mtk::wmma::tcec::store_vector(dst_ptr, frag, layout);
}

template <class Use, int m, int n, int k, class T, class Layout, class Policy>
void store_vector_test() {
	std::printf("!-- %s\n", __func__);
	std::printf("Use    : %s\n", mtk::test_utils::to_string<Use>().c_str());
	std::printf("Layout : %s\n", mtk::test_utils::to_string<Layout>().c_str());
	std::printf("Type   : %s\n", mtk::test_utils::to_string<T>().c_str());
	std::printf("Size   : %u, %u, %u\n", m, n, k);
	std::printf("Policy : Policy<%7s,%9s,%2u,%2u,%2u>\n",
			mtk::test_utils::to_string<typename Policy::op>().c_str(),
			std::is_same<typename Policy::error_correction, mtk::wmma::tcec::with_ec>::value ? "{w/ ec}" : "{w/o ec}",
			Policy::m,
			Policy::n,
			Policy::k);
	constexpr unsigned mem_m = mtk::wmma::tcec::detail::select_value<Use, m, k, m>::value;
	constexpr unsigned mem_n = mtk::wmma::tcec::detail::select_value<Use, k, n, n>::value;

	constexpr auto vec_len = std::is_same<Layout, nvcuda::wmma::col_major>::value ? mem_m : mem_n;

	float* vec_mem;
	float* res_mem;
	float* mat_mem;

	cudaMallocHost(&mat_mem, sizeof(float) * mem_m * mem_n);
	cudaMallocHost(&vec_mem, sizeof(float) * vec_len);
	cudaMallocHost(&res_mem, sizeof(float) * vec_len);

	for (unsigned i = 0; i < vec_len; i++) {
		vec_mem[i] = i;
	}

	for (unsigned i = 0; i < mem_m * mem_n; i++) {
		mat_mem[i] = 0.f;
	}

	for (unsigned i = 0; i < vec_len; i++) {
		mat_mem[i] = vec_mem[i];
	}

	const auto layout = (std::is_same<nvcuda::wmma::col_major, Layout>::value) ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major;
	store_vector_acc_test_kernel<m, n, k, T, Policy><<<1, mtk::test_utils::warp_size>>>(mat_mem, mat_mem, layout);

	cudaDeviceSynchronize();

	float max_error = 0.0f;
	for (unsigned i = 0; i < vec_len; i++) {
		const auto diff = mat_mem[i] - vec_mem[i];
		max_error = std::max(max_error, std::abs(diff));
	}
	std::printf("Error  : %e\n", max_error);

	cudaFree(res_mem);
	cudaFree(vec_mem);
	cudaFree(mat_mem);
}

int main() {
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, half, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();

#ifdef TEST_SIMT
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, float, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, float, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, float, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, float, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, float, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, float, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, float, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, float, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type>();
#endif

#ifdef TEST_TF32
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	load_vector_test<nvcuda::wmma::accumulator , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();
	store_vector_test<nvcuda::wmma::accumulator, 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type>();

	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type>();
	load_vector_test<nvcuda::wmma::matrix_a    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_mma>::type>();
	load_vector_test<nvcuda::wmma::matrix_b    , 32, 32, 32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major, typename mtk::wmma::tcec::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   , mtk::wmma::tcec::op_mma>::type>();
#endif
}

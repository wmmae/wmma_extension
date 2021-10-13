#include <wmma_extension/operators.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include "common.hpp"

//#define TEST_TF32

constexpr unsigned warp_size = 32;

template <class Use, int M, int N, int K, class T, class Layout, class STORAGE_T, class OP_FUNC>
__global__ void test_kernel(
		float* const error,
		const STORAGE_T* const A,
		const STORAGE_T* const B,
		const STORAGE_T* const ref,
		const unsigned ldm,
		const OP_FUNC func
		) {
	nvcuda::wmma::fragment<Use, M, N, K, T, Layout> frag_a, frag_b, frag_ref;
	nvcuda::wmma::load_matrix_sync(frag_a  , A  , ldm);
	nvcuda::wmma::load_matrix_sync(frag_b  , B  , ldm);
	nvcuda::wmma::load_matrix_sync(frag_ref, ref, ldm);

	const auto frag_c = func(frag_a, frag_b);

	float e = 0.;
	for (unsigned i = 0; i < frag_ref.num_elements; i++) {
		const auto diff = std::abs(
				mtk::wmma::detail::common::cast<float>(frag_ref.x[i]) - mtk::wmma::detail::common::cast<float>(frag_c.x[i]));
		e = (e > diff) ? e : diff;
	}
	atomicAdd(error, e);
}

template <int M, int N, int K, class T, class STORAGE_T, class OP_FUNC>
__global__ void test_acc_kernel(
		float* const error,
		const STORAGE_T* const A,
		const STORAGE_T* const B,
		const STORAGE_T* const ref,
		const unsigned ldm,
		const nvcuda::wmma::layout_t layout,
		const OP_FUNC func
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, T> frag_a, frag_b, frag_ref;
	nvcuda::wmma::load_matrix_sync(frag_a  , A  , ldm, layout);
	nvcuda::wmma::load_matrix_sync(frag_b  , B  , ldm, layout);
	nvcuda::wmma::load_matrix_sync(frag_ref, ref, ldm, layout);

	const auto frag_c = func(frag_a, frag_b);

	float e = 0.;
	for (unsigned i = 0; i < frag_ref.num_elements; i++) {
		const auto diff = std::abs(
				mtk::wmma::detail::common::cast<float>(frag_ref.x[i]) - mtk::wmma::detail::common::cast<float>(frag_c.x[i]));
		e = (e > diff) ? e : diff;
	}
	atomicAdd(error, e);
}

template <class Use, int M, int N, int K, class T, class Layout, class storage_t, class OP_FUNC>
struct launch_kernel {
	void operator()(
			float* const error,
			const storage_t* const mat_a,
			const storage_t* const mat_b,
			const storage_t* const mat_r,
			const OP_FUNC op_func
			) {
		test_kernel<Use, M, N, K, T, Layout, storage_t><<<1, warp_size>>>(
				error,
				mat_a, mat_b, mat_r,
				(std::is_same<Layout, nvcuda::wmma::col_major>::value ? mtk::wmma::detail::common::get_M<Use, M, N, K>::value : mtk::wmma::detail::common::get_N<Use, M, N, K>::value),
				op_func
				);
	}
};

template <int M, int N, int K, class T, class Layout, class storage_t, class OP_FUNC>
struct launch_kernel<nvcuda::wmma::accumulator, M, N, K, T, Layout, storage_t, OP_FUNC> {
	void operator()(
			float* const error,
			const storage_t* const mat_a,
			const storage_t* const mat_b,
			const storage_t* const mat_r,
			const OP_FUNC op_func
			) {
		test_acc_kernel<M, N, K, T, storage_t><<<1, warp_size>>>(
				error,
				mat_a, mat_b, mat_r,
				(std::is_same<Layout, nvcuda::wmma::col_major>::value ? mtk::wmma::detail::common::get_M<nvcuda::wmma::accumulator, M, N, K>::value : mtk::wmma::detail::common::get_N<nvcuda::wmma::accumulator, M, N, K>::value),
				(std::is_same<Layout, nvcuda::wmma::col_major>::value ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major),
				op_func
				);
	}
};

template <class Use, int M, int N, int K, class T, class Layout, class OP_FUNC, class ELEM_FUNC>
void test_core(const OP_FUNC op_func, const ELEM_FUNC elem_func, const std::string test_name) {
	const auto num_max_elements = mtk::wmma::detail::common::get_M<Use, M, N, K>::value * mtk::wmma::detail::common::get_N<Use, M, N, K>::value;
	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	storage_t* mat_a;
	storage_t* mat_b;
	storage_t* mat_r;
	cudaMallocHost(&mat_a, num_max_elements * sizeof(storage_t));
	cudaMallocHost(&mat_b, num_max_elements * sizeof(storage_t));
	cudaMallocHost(&mat_r, num_max_elements * sizeof(storage_t));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1, 1);
	for (unsigned i = 0; i < num_max_elements; i++) {
		mat_a[i] = mtk::wmma::detail::common::cast<T>(1.0f);
		mat_b[i] = mtk::wmma::detail::common::cast<T>(1.0f);
		mat_r[i] = elem_func(mat_a[i], mat_b[i]);
	}

	float *error;
	cudaMallocHost(&error, sizeof(float));
	*error = 0.f;
	launch_kernel<Use, M, N, K, T, Layout, storage_t, OP_FUNC>{}(
			error,
			mat_a,
			mat_b,
			mat_r,
			op_func
			);
	cudaDeviceSynchronize();

	std::printf("[operators:%s] %s, <%2d,%2d,%2d>, %s, %s, error = %e, [%s]\n",
			test_name.c_str(),
			mtk::test_utils::get_string<Use>().c_str(),
			M, N, K,
			mtk::test_utils::get_string<T>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str(),
			*error,
			mtk::test_utils::get_test_result_string((*error) < 1e-6)
			);

	cudaFreeHost(error);
	cudaFreeHost(mat_a);
	cudaFreeHost(mat_b);
	cudaFreeHost(mat_r);
}

template <class Use, class Layout>
struct layout_switch {using type = Layout;};
template <class Layout>
struct layout_switch<nvcuda::wmma::accumulator, Layout> {using type = void;};

template <class Use, int M, int N, int K, class T, class Layout>
void test() {
	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	test_core<Use, M, N, K, T, Layout>(
			[]__device__(
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& a,
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& b
				) {return a + b;},
			[](
				const storage_t a,
				const storage_t b
				) {return mtk::wmma::detail::common::cast<float>(a) + mtk::wmma::detail::common::cast<float>(b);},
			"add"
			);
	test_core<Use, M, N, K, T, Layout>(
			[]__device__(
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& a,
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& b
				) {return a - b;},
			[](
				const storage_t a,
				const storage_t b
				) {return mtk::wmma::detail::common::cast<float>(a) - mtk::wmma::detail::common::cast<float>(b);},
			"sub"
			);
	test_core<Use, M, N, K, T, Layout>(
			[]__device__(
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& a,
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>&
				) {return a * mtk::wmma::detail::common::cast<typename nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>::storage_element_type>(3);},
			[](
				const storage_t a,
				const storage_t
				) {return mtk::wmma::detail::common::cast<float>(a) * mtk::wmma::detail::common::cast<float>(3.0f);},
			"mul"
			);
	test_core<Use, M, N, K, T, Layout>(
			[]__device__(
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& a,
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& b
				) {return mtk::wmma::fma(mtk::wmma::detail::common::cast<typename nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>::storage_element_type>(3), a, b);},
			[](
				const storage_t a,
				const storage_t b
				) {return mtk::wmma::detail::common::cast<float>(a) * mtk::wmma::detail::common::cast<float>(3.0f) + mtk::wmma::detail::common::cast<float>(b);},
			"fma"
			);
	test_core<Use, M, N, K, T, Layout>(
			[]__device__(
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>& a,
				const nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>&
				) {return a / mtk::wmma::detail::common::cast<typename nvcuda::wmma::fragment<Use, M, N, K, T, typename layout_switch<Use, Layout>::type>::storage_element_type>(3);},
			[](
				const storage_t a,
				const storage_t
				) {return mtk::wmma::detail::common::cast<float>(a) / mtk::wmma::detail::common::cast<float>(3.0f);},
			"div"
			);
}

int main() {
	test<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 16, half, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::row_major>();
#ifdef TEST_TF32
	test<nvcuda::wmma::matrix_a, 16, 16,  8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b, 16, 16,  8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a, 16, 16,  8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b, 16, 16,  8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();
#endif
}

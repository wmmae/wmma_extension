#include <iostream>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>
#include <wmma_extension/wmma_mma.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

//#define TEST_TF32
//#define TF32_ROUNDING

template <class T, class S>
__device__ __host__ T convert(const S);
template <> __device__ __host__ float convert<float, float>(const float a) {return a;}
template <> __device__ __host__ float convert<float, half >(const half  a) {return __half2float(a);}
template <> __device__ __host__ half  convert<half , float>(const float a) {return __float2half(a);}
template <> __device__ __host__ half  convert<half , half >(const half  a) {return a;}

template <class T>
__device__ T m_abs(const T a) {
	if (a >= convert<T, float>(0)) return a;
	return -a;
}

template <class Use, class Layout>
struct fragment_layout {using type = Layout;};
template <>
struct fragment_layout<nvcuda::wmma::accumulator, nvcuda::wmma::col_major> {using type = void;};
template <>
struct fragment_layout<nvcuda::wmma::accumulator, nvcuda::wmma::row_major> {using type = void;};

template <class T>
__device__ float error_threshold();
template <>
__device__ float error_threshold<float>() {return 1e-6f;};
template <>
__device__ float error_threshold<half >() {return 1e-3f;};

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void test_load_vector_ab_kernel(
		const typename mtk::wmma::detail::common::storage_t<T>::type* const src,
		const typename mtk::wmma::detail::common::storage_t<T>::type* const cor
		) {
	mtk::wmma::mma::fragment<Use, m, n, k, T, Layout> vec_frag;
	mtk::wmma::mma::fill_zero(vec_frag);
	mtk::wmma::mma::load_vector(vec_frag, src);

	mtk::wmma::mma::fragment<Use, m, n, k, T, typename fragment_layout<Use, Layout>::type> cor_frag;
	mtk::wmma::mma::load_matrix_sync(cor_frag, cor, m);

	auto error = convert<typename mtk::wmma::detail::common::storage_t<T>::type, float>(0.0f);
	for (unsigned i = 0; i < vec_frag.num_elements; i++) {
		error += m_abs(vec_frag.x[i] - cor_frag.x[i]);
	}
	printf("[%2u] error = %e (%s)\n",
			threadIdx.x,
			convert<float>(error),
			(convert<float>(error) < error_threshold<T>() ? "PASSED" : "FAILED")
			);
}

template <int m, int n, int k, class T, class Layout>
__global__ void test_load_vector_acc_kernel(
		const T* const src,
		const T* const cor
		) {
	mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, m, n, k, T> vec_frag;
	mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, m, n, k, T> cor_frag;
	mtk::wmma::mma::fill_zero(vec_frag);
	if (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
		mtk::wmma::mma::load_vector(vec_frag, src, nvcuda::wmma::mem_col_major);
		mtk::wmma::mma::load_matrix_sync(cor_frag, cor, m, nvcuda::wmma::mem_col_major);
	} else {
		mtk::wmma::mma::load_vector(vec_frag, src, nvcuda::wmma::mem_row_major);
		mtk::wmma::mma::load_matrix_sync(cor_frag, cor, n, nvcuda::wmma::mem_row_major);
	}

	auto error = convert<typename mtk::wmma::detail::common::storage_t<T>::type, float>(0.0f);
	for (unsigned i = 0; i < vec_frag.num_elements; i++) {
		error += m_abs(vec_frag.x[i] - cor_frag.x[i]);
	}
	printf("[%2u] error = %e (%s)\n",
			threadIdx.x,
			convert<float>(error),
			(convert<float>(error) < error_threshold<T>() ? "PASSED" : "FAILED")
			);
}

template <class Use, int m, int n, int k, class T, class Layout>
void test() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::size_t cor_size = 0;
	std::size_t vec_length = 0;
	std::printf("arch   : %d\n", TEST_ARCH);
	if (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
		std::printf("layout : col_major\n");
	} else if (std::is_same<Layout, nvcuda::wmma::row_major>::value) {
		std::printf("layout : row_major\n");
	} else {
		std::printf("layout : void\n");
	}
	if (std::is_same<float, T>::value)
		std::printf("type   : float\n");
	if (std::is_same<half, T>::value)
		std::printf("type   : half\n");
	if (std::is_same<nvcuda::wmma::precision::tf32, T>::value)
		std::printf("type   : tf32\n");

	if (std::is_same<nvcuda::wmma::matrix_a, Use>::value) {
		std::printf("use    : a\n");
		cor_size = m * k;
		if (std::is_same<nvcuda::wmma::col_major, Layout>::value) {
			vec_length = m;
		} else {
			vec_length = k;
		}
	}
	if (std::is_same<nvcuda::wmma::matrix_b, Use>::value) {
		std::printf("use    : b\n");
		cor_size = n * k;
		if (std::is_same<nvcuda::wmma::col_major, Layout>::value) {
			vec_length = k;
		} else {
			vec_length = n;
		}
	}
	if (std::is_same<nvcuda::wmma::accumulator, Use>::value) {
		std::printf("use    : acc\n");
		cor_size = n * m;
		if (std::is_same<nvcuda::wmma::col_major, Layout>::value) {
			vec_length = m;
		} else {
			vec_length = n;
		}
	}
	std::printf("size   : %d, %d, %d\n", m, n, k);

	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	storage_t* src_mem;
	storage_t* cor_mem;

	cudaMallocHost(&src_mem, m * sizeof(storage_t));
	cudaMallocHost(&cor_mem, cor_size * sizeof(storage_t));

	for (std::size_t i = 0; i < cor_size; i++) {
		cor_mem[i] = convert<storage_t, float>(0);
	}

	for (std::size_t i = 0; i < vec_length; i++) {
		const float v = i * 1.f;
		src_mem[i] = convert<storage_t, float>(v);
		cor_mem[i] = convert<storage_t, float>(v);
	}

	cudaDeviceSynchronize();
	if constexpr (std::is_same<Use, nvcuda::wmma::accumulator>::value) {
		test_load_vector_acc_kernel<m, n, k, T, Layout><<<1, 32>>>(src_mem, cor_mem);
	} else {
		test_load_vector_ab_kernel<Use, m, n, k, T, Layout><<<1, 32>>>(src_mem, cor_mem);
	}
	cudaDeviceSynchronize();
}

int main() {
#if TEST_ARCH >= 80
	test<nvcuda::wmma::matrix_a   , 16, 8, 16, half , nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 16, 8, 16, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 16, float, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 16, float, nvcuda::wmma::row_major>();
#endif

#if TEST_ARCH >= 75
	test<nvcuda::wmma::matrix_a   , 16, 8, 8, half , nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 16, 8, 8, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 8, float, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 8, 8, float, nvcuda::wmma::row_major>();
#endif

#if TEST_ARCH >= 70 && TEST_ARCH <= 75
	test<nvcuda::wmma::matrix_a   , 8, 8, 4, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a   , 8, 8, 4, half , nvcuda::wmma::row_major>();
	test<nvcuda::wmma::matrix_b   , 8, 8, 4, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b   , 8, 8, 4, half , nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 8, 8, 4, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 8, 8, 4, half , nvcuda::wmma::row_major>();
	test<nvcuda::wmma::accumulator, 8, 8, 4, float, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 8, 8, 4, float, nvcuda::wmma::row_major>();
#endif
}

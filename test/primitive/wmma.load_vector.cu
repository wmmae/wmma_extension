#include <iostream>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>
#include "common.hpp"

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

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void test_load_vector_ab_kernel(
		float* const error,
		const typename mtk::wmma::detail::common::storage_t<T>::type* const src,
		const typename mtk::wmma::detail::common::storage_t<T>::type* const cor
		) {
	nvcuda::wmma::fragment<Use, m, n, k, T, Layout> vec_frag;
	mtk::wmma::load_vector(vec_frag, src);

	nvcuda::wmma::fragment<Use, m, n, k, T, typename fragment_layout<Use, Layout>::type> cor_frag;
	nvcuda::wmma::load_matrix_sync(cor_frag, cor, m);

	auto e = convert<typename mtk::wmma::detail::common::storage_t<T>::type, float>(0.0f);
	for (unsigned i = 0; i < vec_frag.num_elements; i++) {
		e += m_abs(vec_frag.x[i] - cor_frag.x[i]);
	}
	if (threadIdx.x == 0) {
		*error = 0;
	}
	__syncthreads();
	for (unsigned i = 0; i < blockDim.x; i++) {
		*error = max(m_abs(e), *error);
		__syncthreads();
	}
}

template <int m, int n, int k, class T, class Layout>
__global__ void test_load_vector_acc_kernel(
		float* const error,
		const T* const src,
		const T* const cor
		) {
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, T> vec_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, T> cor_frag;
	if (std::is_same<Layout, nvcuda::wmma::col_major>::value) {
		mtk::wmma::load_vector(vec_frag, src, nvcuda::wmma::mem_col_major);
		nvcuda::wmma::load_matrix_sync(cor_frag, cor, m, nvcuda::wmma::mem_col_major);
	} else {
		mtk::wmma::load_vector(vec_frag, src, nvcuda::wmma::mem_row_major);
		nvcuda::wmma::load_matrix_sync(cor_frag, cor, m, nvcuda::wmma::mem_row_major);
	}

	auto e = convert<typename mtk::wmma::detail::common::storage_t<T>::type, float>(0.0f);
	for (unsigned i = 0; i < vec_frag.num_elements; i++) {
		e += m_abs(vec_frag.x[i] - cor_frag.x[i]);
	}
	if (threadIdx.x == 0) {
		*error = 0;
	}
	__syncthreads();
	for (unsigned i = 0; i < blockDim.x; i++) {
		*error = max(m_abs(e), *error);
		__syncthreads();
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
void test() {
	std::size_t cor_size = 0;
	std::size_t vec_length = 0;

	if (std::is_same<nvcuda::wmma::matrix_a, Use>::value) {
		cor_size = m * k;
		if (std::is_same<nvcuda::wmma::col_major, Layout>::value) {
			vec_length = m;
		} else {
			vec_length = k;
		}
	}
	if (std::is_same<nvcuda::wmma::matrix_b, Use>::value) {
		cor_size = n * k;
		if (std::is_same<nvcuda::wmma::col_major, Layout>::value) {
			vec_length = k;
		} else {
			vec_length = n;
		}
	}
	if (std::is_same<nvcuda::wmma::accumulator, Use>::value) {
		cor_size = n * m;
		if (std::is_same<nvcuda::wmma::col_major, Layout>::value) {
			vec_length = m;
		} else {
			vec_length = n;
		}
	}

	using storage_t = typename mtk::wmma::detail::common::storage_t<T>::type;
	storage_t* src_mem;
	storage_t* cor_mem;

	cudaMallocHost(&src_mem, m * sizeof(storage_t));
	cudaMallocHost(&cor_mem, cor_size * sizeof(storage_t));

	for (std::size_t i = 0; i < cor_size; i++) {
		cor_mem[i] = convert<storage_t, float>(0);
	}

	for (std::size_t i = 0; i < vec_length; i++) {
		const float v = i / 3.0f;
		src_mem[i] = convert<storage_t, float>(v);
		cor_mem[i] = convert<storage_t, float>(v);
	}

	float* error;
	cudaMallocHost(&error, sizeof(float));
	cudaDeviceSynchronize();
	if constexpr (std::is_same<Use, nvcuda::wmma::accumulator>::value) {
		test_load_vector_acc_kernel<m, n, k, T, Layout><<<1, 32>>>(error, src_mem, cor_mem);
	} else {
		test_load_vector_ab_kernel<Use, m, n, k, T, Layout><<<1, 32>>>(error, src_mem, cor_mem);
	}
	cudaDeviceSynchronize();
	std::printf("[%s] ARCH=%d, <%2d, %2d, %2d>, %10s, %10s, error=%e [%s]\n",
			__FILE__,
			TEST_ARCH,
			m, n, k,
			mtk::test_utils::get_string<T>().c_str(),
			mtk::test_utils::get_string<Layout>().c_str(),
			(*error),
			mtk::test_utils::get_test_result_string((*error) < mtk::test_utils::get_machine_eps<T>() * 16)
			);
	cudaFreeHost(error);
}

int main() {
	test<nvcuda::wmma::matrix_a   , 16, 16, 16, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a   , 16, 16, 16, half , nvcuda::wmma::row_major>();

	test<nvcuda::wmma::matrix_b   , 16, 16, 16, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b   , 16, 16, 16, half , nvcuda::wmma::row_major>();

	test<nvcuda::wmma::accumulator, 16, 16, 16, half , nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::row_major>();
#ifdef TEST_TF32
	test<nvcuda::wmma::matrix_a   , 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_a   , 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();

	test<nvcuda::wmma::matrix_b   , 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::matrix_b   , 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>();

	test<nvcuda::wmma::accumulator, 16, 16, 8, float, nvcuda::wmma::col_major>();
	test<nvcuda::wmma::accumulator, 16, 16, 8, float, nvcuda::wmma::row_major>();
#endif
}

#include <iostream>
#include <type_traits>
#include <random>
#include <mma.h>
#include <wmma_extension/wmma_extension.hpp>
#include <wmma_extension/wmma_mma.hpp>

#ifndef TEST_ARCH
#define TEST_ARCH (-1)
#endif

constexpr unsigned M = 16;
constexpr unsigned N = 8;
constexpr unsigned K = 16;

template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
__global__ void m16n8k16_test_kernel(T* const d, const half* const a, const half* const b, const S* const c) {
	mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a   , M, N, K, half, a_layout> frag_a;
	mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b   , M, N, K, half, b_layout> frag_b;
	mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, T> frag_c;
	mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, M, N, K, S> frag_d;

	const unsigned lda = std::is_same<a_layout, nvcuda::wmma::col_major>::value ? M : K;
	const unsigned ldb = std::is_same<b_layout, nvcuda::wmma::col_major>::value ? K : N;
	const unsigned ldc = (c_layout == nvcuda::wmma::mem_col_major) ? M : N;
	const unsigned ldd = (d_layout == nvcuda::wmma::mem_col_major) ? M : N;

	mtk::wmma::mma::fill_zero(frag_d);

	mtk::wmma::mma::load_matrix_sync(frag_a, a, lda);
	mtk::wmma::mma::load_matrix_sync(frag_b, b, ldb);
	mtk::wmma::mma::load_matrix_sync(frag_c, c, ldc, c_layout);

	mtk::wmma::mma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	mtk::wmma::mma::store_matrix_sync(d, frag_d, ldd, d_layout);
}


template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
double get_residual(const half* const a, const half* const b, const S* const c, const T* const d) {
	double base_norm = 0.0;
	double diff_norm = 0.0;

	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double c_v = 0.0;
			for (unsigned k = 0; k < K; k++) {
				double a_v, b_v;
				if (std::is_same<a_layout, nvcuda::wmma::col_major>::value) {
					a_v = mtk::wmma::detail::common::cast<float>(a[k * M + m]);
				} else {
					a_v = mtk::wmma::detail::common::cast<float>(a[k + K * m]);
				}
				if (std::is_same<b_layout, nvcuda::wmma::col_major>::value) {
					b_v = mtk::wmma::detail::common::cast<float>(b[k + K * n]);
				} else {
					b_v = mtk::wmma::detail::common::cast<float>(b[k * N + n]);
				}
				c_v += a_v * b_v;
			}
			if (c_layout == nvcuda::wmma::mem_col_major) {
				c_v += mtk::wmma::detail::common::cast<float>(c[m + M * n]);
			} else {
				c_v += mtk::wmma::detail::common::cast<float>(c[m * N + n]);
			}

			// compute error
			double d_v;
			if (d_layout == nvcuda::wmma::mem_col_major) {
				d_v = mtk::wmma::detail::common::cast<float>(d[m + M * n]);
			} else {
				d_v = mtk::wmma::detail::common::cast<float>(d[m * N + n]);
			}
			const auto diff = d_v - c_v;

			// accumulate
			diff_norm += diff * diff;
			base_norm += c_v * c_v;
		}
	}
	return std::sqrt(diff_norm / base_norm);
}

template <class T> std::string get_layout_name();
template <> std::string get_layout_name<nvcuda::wmma::col_major>() {return "col";}
template <> std::string get_layout_name<nvcuda::wmma::row_major>() {return "row";}
std::string get_layout_name(const nvcuda::wmma::layout_t layout) {
	if (layout == nvcuda::wmma::mem_col_major) {
		return "col";
	} else {
		return "row";
	}
}

template <class T> std::string get_type_name();
template <> std::string get_type_name<half >() {return "half";}
template <> std::string get_type_name<float>() {return "float";}

template <class T, class S, class a_layout, class b_layout, nvcuda::wmma::layout_t c_layout, nvcuda::wmma::layout_t d_layout>
void test() {
	T* d_ptr;
	S* c_ptr;
	half* a_ptr;
	half* b_ptr;

	cudaMallocHost(&a_ptr, M * K * sizeof(half));
	cudaMallocHost(&b_ptr, K * N * sizeof(half));
	cudaMallocHost(&c_ptr, M * N * sizeof(T));
	cudaMallocHost(&d_ptr, M * N * sizeof(S));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (std::size_t i = 0; i < M * K; i++) {
		a_ptr[i] = mtk::wmma::detail::common::cast<half>(dist(mt));
	}
	for (std::size_t i = 0; i < K * N; i++) {
		b_ptr[i] = mtk::wmma::detail::common::cast<half>(dist(mt));
	}
	for (std::size_t i = 0; i < M * N; i++) {
		c_ptr[i] = mtk::wmma::detail::common::cast<half>(dist(mt));
	}

	cudaDeviceSynchronize();
	m16n8k16_test_kernel<T, S, a_layout, b_layout, c_layout, d_layout><<<1, 32>>>(d_ptr, a_ptr, b_ptr, c_ptr);
	cudaDeviceSynchronize();
	std::printf("[TEST] a_%5s_%s, b_%5s_%s, c_%5s_%s, d_%5s_%s : res = %e\n",
			get_type_name<half>().c_str(), get_layout_name<a_layout>().c_str(),
			get_type_name<half>().c_str(), get_layout_name<b_layout>().c_str(),
			get_type_name<S   >().c_str(), get_layout_name(c_layout).c_str(),
			get_type_name<T   >().c_str(), get_layout_name(d_layout).c_str(),
			get_residual<T, S, a_layout, b_layout, c_layout, d_layout>(a_ptr, b_ptr, c_ptr, d_ptr)
			);
}

#define TEST(c_t, d_t) \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_col_major, nvcuda::wmma::mem_row_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_col_major>(); \
	test<c_t, d_t, nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::mem_row_major, nvcuda::wmma::mem_row_major>(); \

int main() {
	std::printf("-- test (%s) --\n", __FILE__);
	std::printf("arch   : %d\n", TEST_ARCH);

	TEST(float, float);
	TEST(half , half );
}

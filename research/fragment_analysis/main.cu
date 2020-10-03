#include <iostream>
#include <wmma_extension.hpp>

constexpr unsigned warp_size = 32;

template <class Use>
__device__ constexpr unsigned get_num_elements(const int m, const int n, const int k) {return m * n;}
template <>
__device__ constexpr unsigned get_num_elements<nvcuda::wmma::matrix_a>(const int m, const int n, const int k) {return m * k;}
template <>
__device__ constexpr unsigned get_num_elements<nvcuda::wmma::matrix_b>(const int m, const int n, const int k) {return k * n;}

template <class Use, class Layout>
__device__ constexpr unsigned get_ldm(const int m, const int n, const int k);
template <>
__device__ constexpr unsigned get_ldm<nvcuda::wmma::matrix_a, nvcuda::wmma::col_major>(const int m, const int n, const int k) {return m;}
template <>
__device__ constexpr unsigned get_ldm<nvcuda::wmma::matrix_a, nvcuda::wmma::row_major>(const int m, const int n, const int k) {return k;}
template <>
__device__ constexpr unsigned get_ldm<nvcuda::wmma::matrix_b, nvcuda::wmma::col_major>(const int m, const int n, const int k) {return k;}
template <>
__device__ constexpr unsigned get_ldm<nvcuda::wmma::matrix_b, nvcuda::wmma::row_major>(const int m, const int n, const int k) {return n;}

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
template <> std::string get_type_name<__half>() {return "half";}
template <> std::string get_type_name<float >() {return "float";}


template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const nvcuda::wmma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const char* name = "") {
	if ((threadIdx.x & 0x1f) == 0) {
		if (name[0] != '\0') {
			printf("%s = \n", name);
		}
	}
	for (unsigned i = 0; i < warpSize; i++) {
		if (i == (threadIdx.x & 0x1f)) {
			printf("%02u : ", i);
			for (unsigned j = 0; j < frag.num_elements; j++) {
				const auto v = mtk::wmma::detail::common::cast<float>(frag.x[j]);
				if (v == 0.0f) {
					printf("%3.0f ", 0.0f);
				} else if (v > 0) {
					printf("%3.0f ", v);
				}
			}
			printf("\n");
		}
		__syncthreads();
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void ab_fragment_analysis_kernel() {
	constexpr auto num_elements = get_num_elements<Use>(m, n, k);
	__shared__ T mat[num_elements];

	for (unsigned i = 0; i < num_elements; i += warp_size) {
		mat[i + threadIdx.x] = mtk::wmma::detail::common::cast<T>(static_cast<float>(i + threadIdx.x));
	}

	nvcuda::wmma::fragment<Use, m, n, k, T, Layout> frag;
	nvcuda::wmma::load_matrix_sync(frag, mat, get_ldm<Use, Layout>(m, n, k));

	print_fragment(frag);
}

template <int m, int n, int k, class T>
__global__ void c_fragment_analysis_kernel(const nvcuda::wmma::layout_t layout) {
	constexpr auto num_elements = m * n;
	__shared__ T mat[num_elements];

	for (unsigned i = 0; i < num_elements; i += warp_size) {
		mat[i + threadIdx.x] = mtk::wmma::detail::common::cast<T>(static_cast<float>(i + threadIdx.x));
	}

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, T> frag;
	nvcuda::wmma::load_matrix_sync(frag, mat, (layout == nvcuda::wmma::mem_col_major ? m : n), layout);

	print_fragment(frag);
}

template <int m, int n, int k, class T>
void fragment_analysis(
		const std::string use,
		const std::string layout
		) {
	std::printf("# sm_%2d, %15s, (%d, %d, %d), %7s, %10s\n", ARCH, use.c_str(), m, n, k, get_type_name<T>().c_str(), layout.c_str());
	if (use == "accumulator") {
		const auto l = (layout == "col_major" ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major);
		c_fragment_analysis_kernel<m, n, k, T><<<1, warp_size>>>(l);
	} else if (use == "matrix_a") {
		if (layout == "col_major") {
			ab_fragment_analysis_kernel<nvcuda::wmma::matrix_a, m, n, k, T, nvcuda::wmma::col_major><<<1, warp_size>>>();
		} else {
			ab_fragment_analysis_kernel<nvcuda::wmma::matrix_a, m, n, k, T, nvcuda::wmma::row_major><<<1, warp_size>>>();
		}
	} else if (use == "matrix_b") {
		if (layout == "col_major") {
			ab_fragment_analysis_kernel<nvcuda::wmma::matrix_b, m, n, k, T, nvcuda::wmma::col_major><<<1, warp_size>>>();
		} else {
			ab_fragment_analysis_kernel<nvcuda::wmma::matrix_b, m, n, k, T, nvcuda::wmma::row_major><<<1, warp_size>>>();
		}
	}
	cudaDeviceSynchronize();
}

int main() {
	fragment_analysis<16, 16, 16, half>("matrix_a"   , "col_major");
	fragment_analysis<16, 16, 16, half>("matrix_b"   , "col_major");
	fragment_analysis<16, 16, 16, half>("accumulator", "col_major");
	fragment_analysis<16, 16, 16, half>("matrix_a"   , "row_major");
	fragment_analysis<16, 16, 16, half>("matrix_b"   , "row_major");
	fragment_analysis<16, 16, 16, half>("accumulator", "row_major");
}

#include <iostream>
#include <wmma_extension/wmma_extension.hpp>

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

template <class Use>
__device__ constexpr unsigned get_M(const int m, const int n, const int k);
template <>
__device__ constexpr unsigned get_M<nvcuda::wmma::matrix_a>(const int m, const int n, const int k) {return m;}
template <>
__device__ constexpr unsigned get_M<nvcuda::wmma::matrix_b>(const int m, const int n, const int k) {return k;}
template <>
__device__ constexpr unsigned get_M<nvcuda::wmma::accumulator>(const int m, const int n, const int k) {return m;}

template <class Use>
__device__ constexpr unsigned get_N(const int m, const int n, const int k);
template <>
__device__ constexpr unsigned get_N<nvcuda::wmma::matrix_a>(const int m, const int n, const int k) {return k;}
template <>
__device__ constexpr unsigned get_N<nvcuda::wmma::matrix_b>(const int m, const int n, const int k) {return n;}
template <>
__device__ constexpr unsigned get_N<nvcuda::wmma::accumulator>(const int m, const int n, const int k) {return n;}

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
#if ARCH >= 80
template <> std::string get_type_name<nvcuda::wmma::precision::tf32>() {return "tf32";}
#endif

template <class T>
struct get_mem_type {using type = T;};
#if ARCH >= 80
template <>
struct get_mem_type<nvcuda::wmma::precision::tf32> {using type = float;};
#endif

template <class T>
struct c_storage_t {using type = T;};
template <> struct c_storage_t<nvcuda::wmma::precision::tf32> {using type = float;};

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const nvcuda::wmma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const char* name = "") {
	if ((threadIdx.x & 0x1f) == 0) {
		if (name[0] != '\0') {
			printf("%s = \n", name);
		}
	}
	for (unsigned i = 0; i < get_M<MatrixType>(M, N, K); i++) {
		for (unsigned j = 0; j < get_N<MatrixType>(M, N, K); j++) {
			const unsigned element = std::is_same<MemMajor, nvcuda::wmma::col_major>::value ?
				(get_M<MatrixType>(M, N, K) * j + i) : (get_N<MatrixType>(M, N, K) * i + j);
			if (threadIdx.x == 0) {
				printf("(%2u,%2u)-[", i, j);
			}
			__syncwarp();
			for (unsigned k = 0; k < warp_size; k++) {
				if (k == threadIdx.x) {
					for (unsigned l = 0; l < frag.num_elements; l++) {
						if (static_cast<unsigned>(frag.x[l]) == element)
							printf("(%2u,%2u),", threadIdx.x, l);
					}
				}
				__syncwarp();
			}
			if (threadIdx.x == 0) {
				printf("], ");
			}
			__syncwarp();
		}
		if (threadIdx.x == 0) {
			std::printf("\n");
		}
	}
}

template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const nvcuda::wmma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const nvcuda::wmma::layout_t layout, const char* name = "") {
	if ((threadIdx.x & 0x1f) == 0) {
		if (name[0] != '\0') {
			printf("%s = \n", name);
		}
	}
	for (unsigned i = 0; i < get_M<MatrixType>(M, N, K); i++) {
		for (unsigned j = 0; j < get_N<MatrixType>(M, N, K); j++) {
			const unsigned element = (layout == nvcuda::wmma::mem_col_major) ?
				(get_M<MatrixType>(M, N, K) * j + i) : (get_N<MatrixType>(M, N, K) * i + j);
			if (threadIdx.x == 0) {
				printf("(%2u,%2u)-[", i, j);
			}
			__syncwarp();
			for (unsigned k = 0; k < warp_size; k++) {
				if (k == threadIdx.x) {
					for (unsigned l = 0; l < frag.num_elements; l++) {
						if (static_cast<unsigned>(frag.x[l]) == element)
							printf("(%2u,%2u),", threadIdx.x, l);
					}
				}
				__syncwarp();
			}
			if (threadIdx.x == 0) {
				printf("], ");
			}
			__syncwarp();
		}
		if (threadIdx.x == 0) {
			std::printf("\n");
		}
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void ab_fragment_analysis_kernel() {
	using mat_t = typename get_mem_type<T>::type;
	constexpr auto num_elements = get_num_elements<Use>(m, n, k);
	__shared__ mat_t mat[num_elements];

	for (unsigned i = 0; i < num_elements; i += warp_size) {
		mat[i + threadIdx.x] = mtk::wmma::detail::common::cast<mat_t>(static_cast<float>(i + threadIdx.x));
	}

	nvcuda::wmma::fragment<Use, m, n, k, T, Layout> frag;
	nvcuda::wmma::load_matrix_sync(frag, mat, get_ldm<Use, Layout>(m, n, k));

	print_fragment(frag);
}

template <int m, int n, int k, class T>
__global__ void c_fragment_analysis_kernel(const nvcuda::wmma::layout_t layout) {
	using mat_t = typename get_mem_type<T>::type;
	constexpr auto num_elements = m * n;
	__shared__ mat_t mat[num_elements];

	for (unsigned i = 0; i < num_elements; i += warp_size) {
		mat[i + threadIdx.x] = mtk::wmma::detail::common::cast<mat_t>(static_cast<float>(i + threadIdx.x));
	}

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, typename c_storage_t<T>::type> frag;
	nvcuda::wmma::load_matrix_sync(frag, mat, (layout == nvcuda::wmma::mem_col_major ? m : n), layout);

	print_fragment(frag, layout);
}

template <int m, int n, int k, class T>
void ab_fragment_analysis(
		const std::string use,
		const std::string layout
		) {
	std::printf("# sm_%2d, %15s, (%2d, %2d, %2d), %7s, %10s\n", ARCH, use.c_str(), m, n, k, get_type_name<T>().c_str(), layout.c_str());
	if (use == "matrix_a") {
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

template <int m, int n, int k, class T>
void c_fragment_analysis(
		const std::string layout
		) {
	std::printf("# sm_%2d, %15s, (%2d, %2d, %2d), %7s, %10s\n", ARCH, "accumulator", m, n, k, get_type_name<T>().c_str(), layout.c_str());
	const auto l = (layout == "col_major" ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major);
	c_fragment_analysis_kernel<m, n, k, T><<<1, warp_size>>>(l);
	cudaDeviceSynchronize();
}

int main() {
	ab_fragment_analysis<16, 16, 16, half >("matrix_a"   , "col_major");
	ab_fragment_analysis<16, 16, 16, half >("matrix_b"   , "col_major");
	c_fragment_analysis <16, 16, 16, half >("col_major");
	c_fragment_analysis <16, 16, 16, float>("col_major");
	ab_fragment_analysis<16, 16, 16, half >("matrix_a"   , "row_major");
	ab_fragment_analysis<16, 16, 16, half >("matrix_b"   , "row_major");
	c_fragment_analysis <16, 16, 16, half >("row_major");
	c_fragment_analysis <16, 16, 16, float>("row_major");

	ab_fragment_analysis< 8, 32, 16, half >("matrix_a"   , "col_major");
	ab_fragment_analysis< 8, 32, 16, half >("matrix_b"   , "col_major");
	c_fragment_analysis < 8, 32, 16, half >("col_major");
	c_fragment_analysis < 8, 32, 16, float>("col_major");
	ab_fragment_analysis< 8, 32, 16, half >("matrix_a"   , "row_major");
	ab_fragment_analysis< 8, 32, 16, half >("matrix_b"   , "row_major");
	c_fragment_analysis < 8, 32, 16, half >("row_major");
	c_fragment_analysis < 8, 32, 16, float>("row_major");

	ab_fragment_analysis<32,  8, 16, half >("matrix_a"   , "col_major");
	ab_fragment_analysis<32,  8, 16, half >("matrix_b"   , "col_major");
	c_fragment_analysis <32,  8, 16, half >("col_major");
	c_fragment_analysis <32,  8, 16, float>("col_major");
	ab_fragment_analysis<32,  8, 16, half >("matrix_a"   , "row_major");
	ab_fragment_analysis<32,  8, 16, half >("matrix_b"   , "row_major");
	c_fragment_analysis <32,  8, 16, half >("row_major");
	c_fragment_analysis <32,  8, 16, float>("row_major");

#if ARCH >= 80
	ab_fragment_analysis<16, 16,  8, nvcuda::wmma::precision::tf32 >("matrix_a"   , "col_major");
	ab_fragment_analysis<16, 16,  8, nvcuda::wmma::precision::tf32 >("matrix_b"   , "col_major");
	c_fragment_analysis <16, 16,  8, nvcuda::wmma::precision::tf32 >("col_major");
	c_fragment_analysis <16, 16,  8, float>("col_major");
	ab_fragment_analysis<16, 16,  8, nvcuda::wmma::precision::tf32 >("matrix_a"   , "row_major");
	ab_fragment_analysis<16, 16,  8, nvcuda::wmma::precision::tf32 >("matrix_b"   , "row_major");
	c_fragment_analysis <16, 16,  8, nvcuda::wmma::precision::tf32 >("row_major");
	c_fragment_analysis <16, 16,  8, float>("row_major");
#endif
}

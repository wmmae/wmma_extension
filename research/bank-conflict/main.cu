#include <iostream>
#include <wmma_extension/wmma_mma.hpp>

constexpr unsigned bank_size = 32;
constexpr unsigned warp_size = 32;

constexpr unsigned skew = 0;
constexpr unsigned ldm = 64 + skew;

namespace {
template <class T>
std::string get_name_str();
template <> std::string get_name_str<half>() {return "half";}
template <> std::string get_name_str<float>() {return "float";}
template <> std::string get_name_str<nvcuda::wmma::precision::tf32>() {return "tf32";}
template <> std::string get_name_str<nvcuda::wmma::matrix_a>() {return "matrix_a";}
template <> std::string get_name_str<nvcuda::wmma::matrix_b>() {return "matrix_b";}
template <> std::string get_name_str<nvcuda::wmma::accumulator>() {return "accumulator";}
template <> std::string get_name_str<nvcuda::wmma::col_major>() {return "col_major";}
template <> std::string get_name_str<nvcuda::wmma::row_major>() {return "row_major";}
template <> std::string get_name_str<void>() {return "void";}
}

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void kernel(
		unsigned* const bank_array_ptr,
		const nvcuda::wmma::layout_t layout,
		const std::size_t ldm
		) {
	using FRAG_T = mtk::wmma::mma::fragment<Use, m, n, k, T, Layout>;
	FRAG_T frag;

	unsigned frag_i = 0;
	const auto func = [&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned i, const unsigned j) {
		const unsigned mem_index = (layout == nvcuda::wmma::mem_col_major) ? (i + j * ldm) : (j + i * ldm);
		const unsigned bank = mem_index % bank_size;

		atomicAdd(&bank_array_ptr[(frag_i++) * bank_size + bank], 1);

		for (unsigned t = 0; t < bank_size; t++) {
			if (t == threadIdx.x) {
				printf("%3u(%03x) ", bank, mem_index);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			printf("\n");
		}
		__syncthreads();
	};
	if constexpr (std::is_same<Use, nvcuda::wmma::accumulator>::value) {
		mtk::wmma::mma::foreach_ij<FRAG_T>(layout, func);
	} else {
		mtk::wmma::mma::foreach_ij<FRAG_T>(func);
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
void print_bank_conflict(
		const nvcuda::wmma::layout_t layout,
		const std::size_t ldm
		) {
	const unsigned num_elements_per_thread = ((std::is_same<Use, nvcuda::wmma::matrix_a>::value) ? (m * k) : (k * n)) / warp_size;
	unsigned *bank_array;
	cudaMallocHost(&bank_array, sizeof(unsigned) * bank_size * num_elements_per_thread);
	for (unsigned i = 0; i < bank_size * num_elements_per_thread; i++) bank_array[i] = 0;
	std::printf("[<%s,%d,%d,%d,%s,%s>, layout = %s, ldm = %lu] ---------------------------------------------------------- \n",
			get_name_str<Use>().c_str(), m, n, k, get_name_str<T>().c_str(), get_name_str<Layout>().c_str(),
			(layout == nvcuda::wmma::mem_col_major ? "col" : "row"), ldm);
	kernel<Use, m, n, k, T, Layout><<<1, 32>>>(bank_array, layout, ldm);
	cudaDeviceSynchronize();

	for (unsigned i = 0; i < num_elements_per_thread; i++) {
		const auto bank_array_ptr = bank_array + i * bank_size;
		unsigned max_bank_access = 0;
		for (unsigned j = 0; j < bank_size; j++) {
			max_bank_access = std::max(max_bank_access, bank_array_ptr[j]);
		}
		std::printf("[bank_conflict:%2u]: ", max_bank_access - 1);
		for (unsigned j = 0; j < bank_size; j++) {
			std::printf("%2u ", bank_array_ptr[j]);
		}
		std::printf("\n");
	}
	cudaFreeHost(bank_array);
}

int main() {
	print_bank_conflict<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major, ldm);
	print_bank_conflict<nvcuda::wmma::matrix_a, 16, 8, 16, half, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major, ldm);
	print_bank_conflict<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major, ldm);
	print_bank_conflict<nvcuda::wmma::matrix_b, 16, 8, 16, half, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major, ldm);
	print_bank_conflict<nvcuda::wmma::accumulator, 16, 8, 16, float, void>(nvcuda::wmma::mem_col_major, ldm);
	print_bank_conflict<nvcuda::wmma::accumulator, 16, 8, 16, float, void>(nvcuda::wmma::mem_row_major, ldm);

	print_bank_conflict<nvcuda::wmma::matrix_a, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major, ldm);
	print_bank_conflict<nvcuda::wmma::matrix_a, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major, ldm);
	print_bank_conflict<nvcuda::wmma::matrix_b, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major, ldm);
	print_bank_conflict<nvcuda::wmma::matrix_b, 16, 8, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major, ldm);
	print_bank_conflict<nvcuda::wmma::accumulator, 16, 8, 8, float, void>(nvcuda::wmma::mem_col_major, ldm);
	print_bank_conflict<nvcuda::wmma::accumulator, 16, 8, 8, float, void>(nvcuda::wmma::mem_row_major, ldm);
}

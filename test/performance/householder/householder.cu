#include <iostream>
#include <chrono>
#include <wmma_extension/wmma_extension.hpp>

constexpr unsigned warp_size = 32;
constexpr unsigned block_size = 256;
constexpr unsigned test_count = 1024;
constexpr unsigned warp_dim = 16;

template <unsigned DIM>
__device__ void cp_matrix(
		half2* const smem,
		const half2* const gmem
		) {
	for (unsigned i = 0; i < DIM * DIM / 2; i += warp_size) {
		const unsigned index = i + (threadIdx.x & 0x1fu);
		smem[index] = gmem[index];
	}
}

template <unsigned DIM, class HouseholderMatGen>
__global__ void batched_householder_kernel(
		half* const ptr,
		const unsigned batch_size) {
	__shared__ half smem_mat[DIM * DIM * block_size / warp_size];
	__shared__ half smem_vec[DIM * block_size / warp_size];

	half* const smem_mat_ptr = smem_mat + DIM * DIM * (threadIdx.x / warp_size);
	half* const smem_vec_ptr = smem_vec + DIM * (threadIdx.x / warp_size);

	const unsigned matrix_id = threadIdx.x + blockIdx.x * blockDim.x / warp_size;
	if (matrix_id >= batch_size) return;

	cp_matrix<warp_dim>(
			reinterpret_cast<half2*>(smem_mat_ptr),
			reinterpret_cast<half2*>(ptr + DIM * DIM * ((threadIdx.x + block_size / warp_size * blockIdx.x) / warp_size))
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, warp_dim, warp_dim, warp_dim, half, nvcuda::wmma::col_major> frag_b[warp_dim * warp_dim / (warp_dim * warp_dim)];
	for (unsigned i = 0; i < DIM / warp_dim; i += 1) {
		for (unsigned j = 0; j < DIM / warp_dim; j += 1) {
			nvcuda::wmma::load_matrix_sync(frag_b[i + j * (DIM / warp_dim)], smem_mat_ptr + j * warp_dim + DIM * warp_dim * i, DIM);
		}
	}

	if ((threadIdx.x & 0x1f) < DIM) {
		smem_vec_ptr[(threadIdx.x & 0x1f)] = smem_mat_ptr[(threadIdx.x & 0x1f)];
	}
	__syncwarp();

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, warp_dim, warp_dim, warp_dim, half, nvcuda::wmma::col_major> frag_a[warp_dim * warp_dim / (warp_dim * warp_dim)];
	HouseholderMatGen{}(frag_a, smem_mat_ptr, smem_vec_ptr);



	for (unsigned i = 0; i < DIM / warp_dim; i += 1) {
		for (unsigned j = 0; j < DIM / warp_dim; j += 1) {
			nvcuda::wmma::fragment<nvcuda::wmma::accumulator, warp_dim, warp_dim, warp_dim, half> frag_c;
			nvcuda::wmma::fill_fragment(frag_c, 0.f);
			for (unsigned k = 0; k < DIM / warp_dim; k += 1) {
				nvcuda::wmma::mma_sync(frag_c, frag_a[i + k * (DIM / warp_dim)], frag_b[k + j * (DIM / warp_dim)], frag_c);
			}
			nvcuda::wmma::store_matrix_sync(smem_mat_ptr + i * warp_dim + j * warp_dim * DIM, frag_c, DIM, nvcuda::wmma::mem_col_major);
		}
	}
	cp_matrix<DIM>(
			reinterpret_cast<half2*>(ptr + DIM * DIM * ((threadIdx.x + block_size / warp_size * blockIdx.x) / warp_size)),
			reinterpret_cast<half2*>(smem_mat_ptr)
			);
}

template <unsigned DIM>
struct HouseholderMatGenWMMA {
	__device__ void operator()(
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, warp_dim, warp_dim, warp_dim, half, nvcuda::wmma::col_major>* frag,
			half* const smem_mat,
			half* const smem_vec
			) const {
#pragma unroll
		for (unsigned i = 0; i < DIM * DIM; i += warp_size) {
			const unsigned index = i + (threadIdx.x & 0x1fu);
			const unsigned m = index % DIM;
			const unsigned n = index / DIM;

			half v = smem_vec[m] * smem_vec[n] * __float2half(-2.f);
			if (m == n) {
				v += __float2half(1.f);
			}
			__syncwarp();
			smem_mat[index] = v;
		}
		for (unsigned i = 0; i < DIM / warp_dim; i += 1) {
			for (unsigned j = 0; j < DIM / warp_dim; j += 1) {
				nvcuda::wmma::load_matrix_sync(frag[i + j * (DIM / warp_dim)], smem_mat + i * warp_dim + DIM * warp_dim * j, DIM);
			}
		}
	}
};

template <unsigned DIM>
struct HouseholderMatGenWMMAe {
	__device__ void operator()(
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, warp_dim, warp_dim, warp_dim, half, nvcuda::wmma::col_major>* frag,
			half* const,
			const half* const smem_vec
			) const {
		mtk::wmma::foreach_ij<nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, warp_dim, warp_dim, warp_dim, half, nvcuda::wmma::col_major>>(
				[&](const unsigned *list, const unsigned list_size, const unsigned m, const unsigned n) {
					for (unsigned i = 0; i < DIM; i += warp_dim) {
						for (unsigned j = 0; j < DIM; j += warp_dim) {
							half v = smem_vec[i + m] * smem_vec[j + n] * __float2half(-2.f);
							if (m == n && j == i) {
								v += __float2half(1.f);
							}
							__syncwarp();
#pragma unroll
							for (unsigned f = 0; f < list_size; f++) {
								frag[i / warp_dim + j / warp_dim * (DIM / warp_dim)].x[f] = v;
							}
						}
					}
				});
		__syncwarp();
	}
};

template <class T>
std::string get_class_name();
template <> std::string get_class_name<HouseholderMatGenWMMA <16>>() {return "wmma_16";}
template <> std::string get_class_name<HouseholderMatGenWMMAe<16>>() {return "wmmae_16";}
template <> std::string get_class_name<HouseholderMatGenWMMA <32>>() {return "wmma_32";}
template <> std::string get_class_name<HouseholderMatGenWMMAe<32>>() {return "wmmae_32";}

template <unsigned DIM, class HouseholderMatGen>
void batched_householder(
		half* const ptr,
		const unsigned batch_size
		) {
	const unsigned grid_size = (batch_size * warp_size + block_size - 1) / block_size;
	batched_householder_kernel<DIM, HouseholderMatGen><<<grid_size, block_size>>>(ptr, batch_size);
}

template <unsigned DIM, class HouseholderMatGen>
void test_batched_kernel(
		const unsigned batch_size
		) {
	half* input_matrix;
	cudaMalloc(&input_matrix, sizeof(half) * DIM * DIM * batch_size);
	const auto start_clock = std::chrono::system_clock::now();
	for (unsigned c = 0; c < test_count; c++) {
		batched_householder<DIM, HouseholderMatGen>(
				input_matrix,
				batch_size);
	}
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();
	cudaFree(input_matrix);

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / static_cast<double>(test_count) * 1e-6;

	std::printf("%u,%s,%e\n", batch_size, get_class_name<HouseholderMatGen>().c_str(), elapsed_time);
}

int main() {
	std::printf("batch_size,api,time\n");
	for (unsigned i = 13; i <= 21; i++) {
		test_batched_kernel<32, HouseholderMatGenWMMA <32>>(1u << i);
		test_batched_kernel<32, HouseholderMatGenWMMAe<32>>(1u << i);
		test_batched_kernel<16, HouseholderMatGenWMMA <16>>(1u << i);
		test_batched_kernel<16, HouseholderMatGenWMMAe<16>>(1u << i);
	}
}

#include <iostream>
#include <chrono>
#include <wmma_extension/wmma_extension.hpp>

constexpr unsigned warp_size = 32;
constexpr unsigned block_size = 256;
constexpr unsigned test_count = 1u << 12;

template <unsigned DIM>
__device__ void cp_matrix(
		half2* const smem,
		const half2* const gmem
		) {
	for (unsigned i = 0; i < DIM * DIM / 2; i += warp_size * 4) {
		const unsigned index = i + (threadIdx.x & 0x1fu);
		reinterpret_cast<uint4*>(smem)[index] = reinterpret_cast<const uint4*>(gmem)[index];
	}
}

template <unsigned DIM, class GivensMatGen>
__global__ void batched_givens_kernel(
		half* const ptr,
		const unsigned batch_size) {
	__shared__ half smem[DIM * DIM * block_size / warp_size];

	const unsigned gi = 5;
	const unsigned gj = 6;
	const float theta = M_PI / 6;
	half* const smem_ptr = smem + DIM * DIM * (threadIdx.x / warp_size);

	const unsigned matrix_id = threadIdx.x + blockIdx.x * blockDim.x / warp_size;
	if (matrix_id >= batch_size) return;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, DIM, DIM, DIM, half, nvcuda::wmma::col_major> frag_a;
	cp_matrix<DIM>(
			reinterpret_cast<half2*>(smem_ptr),
			reinterpret_cast<half2*>(ptr + DIM * DIM * ((threadIdx.x + block_size / warp_size * blockIdx.x) / warp_size))
			);
	GivensMatGen{}(frag_a, gi, gj, theta, smem_ptr);
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, DIM, DIM, DIM, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::load_matrix_sync(frag_b, smem_ptr, DIM);

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, DIM, DIM, DIM, half> frag_c;
	nvcuda::wmma::fill_fragment(frag_c, 0.f);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(smem_ptr, frag_c, DIM, nvcuda::wmma::mem_col_major);
	cp_matrix<DIM>(
			reinterpret_cast<half2*>(ptr + DIM * DIM * ((threadIdx.x + block_size / warp_size * blockIdx.x) / warp_size)),
			reinterpret_cast<half2*>(smem_ptr)
			);
}

template <unsigned DIM, class GivensMatGen>
__global__ void batched_givens_kernel(
		half* const ptr,
		const unsigned gi, const unsigned gj,
		const unsigned batch_size) {
	__shared__ half smem[DIM * DIM * block_size / warp_size];

	const float theta = M_PI / 6;
	half* const smem_ptr = smem + DIM * DIM * (threadIdx.x / warp_size);

	const unsigned matrix_id = threadIdx.x + blockIdx.x * blockDim.x / warp_size;
	if (matrix_id >= batch_size) return;

	cp_matrix<DIM>(
			reinterpret_cast<half2*>(smem_ptr),
			reinterpret_cast<half2*>(ptr + DIM * DIM * ((threadIdx.x + block_size / warp_size * blockIdx.x) / warp_size))
			);
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, DIM, DIM, DIM, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::load_matrix_sync(frag_b, smem_ptr, DIM);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, DIM, DIM, DIM, half, nvcuda::wmma::col_major> frag_a;
	GivensMatGen{}(frag_a, gi, gj, theta, smem_ptr);

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, DIM, DIM, DIM, half> frag_c;
	nvcuda::wmma::fill_fragment(frag_c, 0.f);

	nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

	nvcuda::wmma::store_matrix_sync(smem_ptr, frag_c, DIM, nvcuda::wmma::mem_col_major);
	cp_matrix<DIM>(
			reinterpret_cast<half2*>(ptr + DIM * DIM * ((threadIdx.x + block_size / warp_size * blockIdx.x) / warp_size)),
			reinterpret_cast<half2*>(smem_ptr)
			);
}

template <unsigned DIM>
struct GivensMatGenWMMA {
	__device__ GivensMatGenWMMA(){}
	__device__ void operator()(
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, DIM, DIM, DIM, half, nvcuda::wmma::col_major>& frag,
			const unsigned gi,
			const unsigned gj,
			const float theta,
			half* const smem
			) const {
		half2* smem_h2_ptr = reinterpret_cast<half2*>(smem);
		for (unsigned i = 0; i < DIM * DIM / 2; i += warp_size) {
			const unsigned index = i + (threadIdx.x & 0x1fu);
			*reinterpret_cast<uint32_t*>(&(smem_h2_ptr[index])) = 0u;
		}
		__syncwarp();
		const auto lane_id = (threadIdx.x & 0x1f);
		if (lane_id < 16) {
			smem[(1 + DIM) * lane_id] = __float2half(1.f);
		}
		__syncwarp();
		if (lane_id == 0) {
			const auto c = __float2half(__cosf(theta));
			smem[gi + gi * DIM] = c;
			smem[gj + gj * DIM] = c;
			const auto s = __float2half(__sinf(theta));
			smem[gi + gj * DIM] =-s;
			smem[gj + gi * DIM] = s;
		}
		__syncwarp();
		nvcuda::wmma::load_matrix_sync(frag, smem, DIM);
	}
};

template <unsigned DIM>
struct GivensMatGenWMMAe {
	__device__ GivensMatGenWMMAe(){}
	__device__ void operator()(
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, DIM, DIM, DIM, half, nvcuda::wmma::col_major>& frag,
			const unsigned gi,
			const unsigned gj,
			const float theta,
			half* const
			) const {
		mtk::wmma::fill_zero(frag);
		unsigned tid_list[2];
		unsigned fid_list[2];
		unsigned list_size;
		for (unsigned i = 0; i < DIM; i++) {
			mtk::wmma::map<decltype(frag)>(tid_list, fid_list, list_size, i, i);
			for (unsigned k = 0; k < list_size; k++) {
				if (threadIdx.x == tid_list[k]) {
					frag.x[fid_list[k]] = 1.0f;
				}
			}
		}
		const auto c = __float2half(__cosf(theta));
		mtk::wmma::map<decltype(frag)>(tid_list, fid_list, list_size, gi, gi);
		for (unsigned k = 0; k < list_size; k++) {
			if (threadIdx.x == tid_list[k]) {
				frag.x[fid_list[k]] = c;
			}
		}
		mtk::wmma::map<decltype(frag)>(tid_list, fid_list, list_size, gj, gj);
		for (unsigned k = 0; k < list_size; k++) {
			if (threadIdx.x == tid_list[k]) {
				frag.x[fid_list[k]] = c;
			}
		}
		const auto s = __float2half(__sinf(theta));
		mtk::wmma::map<decltype(frag)>(tid_list, fid_list, list_size, gi, gj);
		for (unsigned k = 0; k < list_size; k++) {
			if (threadIdx.x == tid_list[k]) {
				frag.x[fid_list[k]] = -s;
			}
		}
		mtk::wmma::map<decltype(frag)>(tid_list, fid_list, list_size, gj, gi);
		for (unsigned k = 0; k < list_size; k++) {
			if (threadIdx.x == tid_list[k]) {
				frag.x[fid_list[k]] = s;
			}
		}
	}
};

template <class T>
std::string get_class_name();
template <> std::string get_class_name<GivensMatGenWMMA <16>>() {return "wmma_16";}
template <> std::string get_class_name<GivensMatGenWMMAe<16>>() {return "wmmae_16";}

template <unsigned DIM, class GivensMatGen>
void batched_givens(
		half* const ptr,
		const unsigned batch_size
		) {
	const unsigned grid_size = (batch_size * warp_size + block_size - 1) / block_size;
	batched_givens_kernel<DIM, GivensMatGen><<<grid_size, block_size>>>(ptr, batch_size);
}

template <unsigned DIM, class GivensMatGen>
void test_batched_kernel(
		const unsigned batch_size
		) {
	half* input_matrix;
	cudaMalloc(&input_matrix, sizeof(half) * DIM * DIM * batch_size);
	const auto start_clock = std::chrono::system_clock::now();
	for (unsigned c = 0; c < test_count; c++) {
		batched_givens<DIM, GivensMatGen>(
				input_matrix,
				batch_size);
	}
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();
	cudaFree(input_matrix);

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / static_cast<double>(test_count) * 1e-6;

	std::printf("%u,1,%s,%e\n", batch_size, get_class_name<GivensMatGen>().c_str(), elapsed_time);
}

template <unsigned DIM, class GivensMatGen>
void batched_givens(
		half* const ptr,
		const unsigned gi, const unsigned gj,
		const unsigned batch_size
		) {
	const unsigned grid_size = (batch_size * warp_size + block_size - 1) / block_size;
	batched_givens_kernel<DIM, GivensMatGen><<<grid_size, block_size>>>(ptr, gi, gj, batch_size);
}

template <unsigned DIM, class GivensMatGen>
void test_batched_kernel(
		const unsigned gi, const unsigned gj,
		const unsigned batch_size
		) {
	half* input_matrix;
	cudaMalloc(&input_matrix, sizeof(half) * DIM * DIM * batch_size);
	const auto start_clock = std::chrono::system_clock::now();
	for (unsigned c = 0; c < test_count; c++) {
		batched_givens<DIM, GivensMatGen>(
				input_matrix,
				gi, gj,
				batch_size);
	}
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();
	cudaFree(input_matrix);

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / static_cast<double>(test_count) * 1e-6;

	std::printf("%u,0,%s,%e\n", batch_size, get_class_name<GivensMatGen>().c_str(), elapsed_time);
}

int main() {
	std::printf("batch_size,embedded,api,time\n");
	for (unsigned i = 13; i <= 23; i++) {
		test_batched_kernel<16, GivensMatGenWMMA <16>>(1u << i);
		test_batched_kernel<16, GivensMatGenWMMAe<16>>(1u << i);
		test_batched_kernel<16, GivensMatGenWMMA <16>>(5, 6, 1u << i);
		test_batched_kernel<16, GivensMatGenWMMAe<16>>(5, 6, 1u << i);
	}
}

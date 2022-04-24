#include <iostream>
#include <wmma_extension/utils.hpp>

template <class T>
__host__ __device__ constexpr unsigned get_size_in_byte();
template <> __host__ __device__ constexpr unsigned get_size_in_byte<float >() {return 4;};
template <> __host__ __device__ constexpr unsigned get_size_in_byte<float2>() {return 8;};
template <> __host__ __device__ constexpr unsigned get_size_in_byte<float4>() {return 16;};

template <class T, unsigned block_size>
__global__ void cp_async_test_kernel(
		T* const dst_ptr,
		const T* const src_ptr
		) {
	__shared__ T smem[block_size];

	mtk::wmma::utils::cp_async::cp_async<get_size_in_byte<T>()>(smem + threadIdx.x, src_ptr + threadIdx.x);
	mtk::wmma::utils::cp_async::commit();

	mtk::wmma::utils::cp_async::wait_all();
	dst_ptr[threadIdx.x] = smem[threadIdx.x];
}

template <class T, unsigned block_size>
void cp_async_test() {
	T* d_input;
	T* d_output;
	T* h_input;
	T* h_output;

	cudaMalloc(&d_input, sizeof(T) * block_size);
	cudaMalloc(&d_output, sizeof(T) * block_size);
	cudaMallocHost(&h_input, sizeof(T) * block_size);
	cudaMallocHost(&h_output, sizeof(T) * block_size);

	for (unsigned i = 0; i < block_size * get_size_in_byte<T>() / 4; i++) {
		reinterpret_cast<float*>(h_input)[i] = i;
	}

	cudaMemcpy(d_input, h_input, block_size * sizeof(T), cudaMemcpyDefault);

	cp_async_test_kernel<T, block_size><<<1, block_size>>>(d_output, d_input);

	cudaMemcpy(h_output, d_output, block_size * sizeof(T), cudaMemcpyDefault);

	double max_error = 0;
	for (unsigned i = 0; i < block_size * get_size_in_byte<T>() / 4; i++) {
		const double diff = reinterpret_cast<float*>(h_output)[i] - reinterpret_cast<float*>(h_input)[i];
		max_error = std::max(std::abs(diff), max_error);
	}

	std::printf("%s[%2u Byte] error = %e\n", __func__, get_size_in_byte<T>(), max_error);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_input);
	cudaFreeHost(h_output);
}

int main() {
	cp_async_test<float , 128>();
	cp_async_test<float2, 128>();
	cp_async_test<float4, 128>();
}


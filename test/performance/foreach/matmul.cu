#include <chrono>
#include <iostream>
#include <mma.h>
#include <type_traits>
#include <wmma_extension/wmma_extension.hpp>

constexpr unsigned block_size = 256;
constexpr unsigned warp_size = 32;

#ifndef CUDA_ARCH_SM
#define CUDA_ARCH_SM 0
#endif

// Copy a SMEM_M-by-SMEM_N matrix on the device memory to shared memory
template <unsigned SMEM_M, unsigned SMEM_N>
__device__ void d2s_col_cta(const float *const d_ptr, float *const f32_s_ptr,
                            half *const f16_s_ptr, const unsigned ld) {
  for (unsigned i = threadIdx.x; i < SMEM_M * SMEM_N; i += block_size) {
    const auto im = i % SMEM_M;
    const auto in = i / SMEM_M;
    const float v = d_ptr[im + in * ld];
    f32_s_ptr[i] = v;
    f16_s_ptr[i] = __float2half(v);
  }
}

template <bool UseWMMAe>
__global__ void matmul(float *const c_ptr, const float *const a_ptr,
                       const float *const b_ptr, const unsigned n);

template <>
__global__ void matmul<true>(float *const c_ptr, const float *const a_ptr,
                             const float *const b_ptr, const unsigned n) {
  constexpr unsigned FDIM = 16;
  __shared__ float F32_smem[block_size * warp_size];
  __shared__ half F16_smem[block_size * warp_size];

  const unsigned lane_id = threadIdx.x & 0x1f;
  const unsigned warp_id = threadIdx.x >> 5;

  const unsigned block_c_row = (blockIdx.x % (n / warp_size)) * warp_size;
  const unsigned block_c_col = (blockIdx.x / (n / warp_size)) * warp_size;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float>
      frag_c[4];
  for (unsigned i = 0; i < 4; i++) {
    nvcuda::wmma::fill_fragment(frag_c[i], 0.0f);
  }

  for (unsigned kb = 0; kb < n; kb += block_size) {
    d2s_col_cta<warp_size, block_size>(a_ptr + block_c_row + kb * n, F32_smem,
                                       F16_smem, n);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half,
                           nvcuda::wmma::col_major>
        frag_a[4], frag_da[4];
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        unsigned long offset =
            warp_id * warp_size * warp_size + i * FDIM + j * FDIM * warp_size;
        nvcuda::wmma::load_matrix_sync(frag_a[i + 2 * j], F16_smem + offset,
                                       warp_size);
      }
    }
    const float *matrix_a_block_head =
        F32_smem + warp_id * warp_size * warp_size;
    mtk::wmma::foreach<decltype(frag_a[0])>(
        [&](const unsigned *frag_index_list,
            const unsigned fragment_index_count, const unsigned mem_index) {
          for (unsigned i = 0; i < fragment_index_count; i++) {
            const auto f_index = frag_index_list[i];
            frag_da[0].x[f_index] =
                __float2half(matrix_a_block_head[mem_index] -
                             __half2float(frag_a[0].x[f_index]));
            frag_da[1].x[f_index] =
                __float2half(matrix_a_block_head[mem_index + FDIM] -
                             __half2float(frag_a[1].x[f_index]));
            frag_da[2].x[f_index] =
                __float2half(matrix_a_block_head[mem_index + FDIM * warp_size] -
                             __half2float(frag_a[2].x[f_index]));
            frag_da[3].x[f_index] = __float2half(
                matrix_a_block_head[mem_index + FDIM * warp_size + FDIM] -
                __half2float(frag_a[3].x[f_index]));
          }
        });

    __syncthreads();
    d2s_col_cta<block_size, warp_size>(b_ptr + kb + block_c_col * n, F32_smem,
                                       F16_smem, n);

    __syncthreads();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half,
                           nvcuda::wmma::col_major>
        frag_b[4], frag_db[4];
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        unsigned long offset =
            warp_id * warp_size + i * FDIM + j * FDIM * block_size;
        nvcuda::wmma::load_matrix_sync(frag_b[i + 2 * j], F16_smem + offset,
                                       block_size);
      }
    }
    const float *matrix_b_block_head = F32_smem + warp_id * warp_size;
    mtk::wmma::foreach<decltype(frag_b[0])>(
        [&](const unsigned *frag_index_list,
            const unsigned fragment_index_count, const unsigned mem_index) {
          for (unsigned i = 0; i < fragment_index_count; i++) {
            const auto f_index = frag_index_list[i];
            const unsigned r = mem_index & 0xf;
            const unsigned c = mem_index >> 4;
            const unsigned long j = r + c * block_size;
            frag_db[0].x[f_index] = __float2half(
                matrix_b_block_head[j] - __half2float(frag_b[0].x[f_index]));
            frag_db[1].x[f_index] =
                __float2half(matrix_b_block_head[j + FDIM] -
                             __half2float(frag_b[1].x[f_index]));
            frag_db[2].x[f_index] =
                __float2half(matrix_b_block_head[j + FDIM * block_size] -
                             __half2float(frag_b[2].x[f_index]));
            frag_db[3].x[f_index] =
                __float2half(matrix_b_block_head[j + FDIM * block_size + FDIM] -
                             __half2float(frag_b[3].x[f_index]));
          }
        });

    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        const unsigned c_index = i + j * 2;
        for (unsigned k = 0; k < 2; k++) {
          const unsigned a_index = i + k * 2;
          const unsigned b_index = j * 2 + k;

          nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index],
                                 frag_b[b_index], frag_c[c_index]);
          nvcuda::wmma::mma_sync(frag_c[c_index], frag_da[a_index],
                                 frag_b[b_index], frag_c[c_index]);
          nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index],
                                 frag_db[b_index], frag_c[c_index]);
        }
      }
    }
  }
  for (unsigned i = 0; i < 2; i++) {
    for (unsigned j = 0; j < 2; j++) {
      const unsigned c_index = i + j * 2;
      nvcuda::wmma::store_matrix_sync(
          F32_smem + warp_size * warp_size * warp_id + i * FDIM +
              j * warp_size * FDIM,
          frag_c[c_index], warp_size, nvcuda::wmma::mem_col_major);
    }
  }
  __syncthreads();
  for (unsigned i = 0; i < warp_size * warp_size; i += block_size) {
    float v = 0.0f;
    for (unsigned j = 0; j < (block_size / warp_size); j++) {
      v += F32_smem[i + threadIdx.x + j * warp_size * warp_size];
    }
    c_ptr[(block_c_col + warp_id) * n + block_c_row + lane_id] = v;
  }
}

template <>
__global__ void matmul<false>(float *const c_ptr, const float *const a_ptr,
                              const float *const b_ptr, const unsigned n) {
  constexpr unsigned FDIM = 16;
  __shared__ float F32_smem[block_size * warp_size];
  __shared__ half F16_smem[block_size * warp_size];

  const unsigned lane_id = threadIdx.x & 0x1f;
  const unsigned warp_id = threadIdx.x >> 5;

  const unsigned block_c_row = (blockIdx.x % (n / warp_size)) * warp_size;
  const unsigned block_c_col = (blockIdx.x / (n / warp_size)) * warp_size;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FDIM, FDIM, FDIM, float>
      frag_c[4];
  for (unsigned i = 0; i < 4; i++) {
    nvcuda::wmma::fill_fragment(frag_c[i], 0.0f);
  }

  for (unsigned kb = 0; kb < n; kb += block_size) {
    d2s_col_cta<warp_size, block_size>(a_ptr + block_c_row + kb * n, F32_smem,
                                       F16_smem, n);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FDIM, FDIM, FDIM, half,
                           nvcuda::wmma::col_major>
        frag_a[4], frag_da[4];
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        unsigned long offset =
            warp_id * warp_size * warp_size + i * FDIM + j * FDIM * warp_size;
        nvcuda::wmma::load_matrix_sync(frag_a[i + 2 * j], F16_smem + offset,
                                       warp_size);
      }
    }
    __syncthreads();
    for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
      const unsigned smem_index = threadIdx.x + i;
      F16_smem[smem_index] = __float2half(F32_smem[smem_index] -
                                          __half2float(F16_smem[smem_index]));
    }
    __syncthreads();
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        unsigned long offset =
            warp_id * warp_size + i * FDIM + j * FDIM * block_size;
        nvcuda::wmma::load_matrix_sync(frag_da[i + 2 * j], F16_smem + offset,
                                       block_size);
      }
    }
    __syncthreads();

    d2s_col_cta<block_size, warp_size>(b_ptr + kb + block_c_col * n, F32_smem,
                                       F16_smem, n);
    __syncthreads();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FDIM, FDIM, FDIM, half,
                           nvcuda::wmma::col_major>
        frag_b[4], frag_db[4];
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        unsigned long offset =
            warp_id * warp_size + i * FDIM + j * FDIM * block_size;
        nvcuda::wmma::load_matrix_sync(frag_b[i + 2 * j], F16_smem + offset,
                                       block_size);
      }
    }
    __syncthreads();
    for (unsigned i = 0; i < block_size * warp_size; i += block_size) {
      const unsigned smem_index = threadIdx.x + i;
      F16_smem[smem_index] = __float2half(F32_smem[smem_index] -
                                          __half2float(F16_smem[smem_index]));
    }
    __syncthreads();
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        unsigned long offset =
            warp_id * warp_size + i * FDIM + j * FDIM * block_size;
        nvcuda::wmma::load_matrix_sync(frag_db[i + 2 * j], F16_smem + offset,
                                       block_size);
      }
    }

    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        const unsigned c_index = i + j * 2;
        for (unsigned k = 0; k < 2; k++) {
          const unsigned a_index = i + k * 2;
          const unsigned b_index = j * 2 + k;

          nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index],
                                 frag_b[b_index], frag_c[c_index]);
          nvcuda::wmma::mma_sync(frag_c[c_index], frag_da[a_index],
                                 frag_b[b_index], frag_c[c_index]);
          nvcuda::wmma::mma_sync(frag_c[c_index], frag_a[a_index],
                                 frag_db[b_index], frag_c[c_index]);
        }
      }
    }
  }
  for (unsigned i = 0; i < 2; i++) {
    for (unsigned j = 0; j < 2; j++) {
      const unsigned c_index = i + j * 2;
      nvcuda::wmma::store_matrix_sync(
          F32_smem + warp_size * warp_size * warp_id + i * FDIM +
              j * warp_size * FDIM,
          frag_c[c_index], warp_size, nvcuda::wmma::mem_col_major);
    }
  }
  __syncthreads();
  for (unsigned i = 0; i < warp_size * warp_size; i += block_size) {
    float v = 0.0f;
    for (unsigned j = 0; j < (block_size / warp_size); j++) {
      v += F32_smem[i + threadIdx.x + j * warp_size * warp_size];
    }
    c_ptr[(block_c_col + warp_id) * n + block_c_row + lane_id] = v;
  }
}

template <bool UseWMMAe> void test_matmul(const unsigned size_power) {
  constexpr std::size_t C = 1lu << 6;
  const std::size_t N = 1lu << size_power;
  const std::size_t grid_size = (N / warp_size) * (N / warp_size);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * N * N);
  cudaMalloc(&dB, sizeof(float) * N * N);
  cudaMalloc(&dC, sizeof(float) * N * N);

  const auto start_clock = std::chrono::system_clock::now();
  for (std::size_t c = 0; c < C; c++) {
    matmul<UseWMMAe><<<grid_size, block_size>>>(dC, dA, dB, N);
  }
  const auto status = cudaGetLastError();
  cudaDeviceSynchronize();
  if (status != 0) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
  }
  const auto end_clock = std::chrono::system_clock::now();
  const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_clock -
                                                            start_clock)
          .count() /
      1.e6 / C;

  std::printf("%u,%lu,%u,%e,%e\n", static_cast<unsigned>(CUDA_ARCH_SM), N,
              (UseWMMAe ? 1u : 0u), elapsed_time,
              (2 * N * N * N) / elapsed_time / (1lu << 40));

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

void print_header() { std::printf("arch,size,wmmae,time,performance\n"); }

void test_matmul(const unsigned min_p, const unsigned max_p) {
  std::printf("# %s\n", __func__);
  std::printf("-- 1\n");
  print_header();
  for (unsigned i = min_p; i <= max_p; i++) {
    test_matmul<false>(i);
  }
  for (unsigned i = min_p; i <= max_p; i++) {
    test_matmul<true>(i);
  }
  std::printf("-- 2\n");
  print_header();
  for (unsigned i = min_p; i <= max_p; i++) {
    test_matmul<false>(i);
  }
  for (unsigned i = min_p; i <= max_p; i++) {
    test_matmul<true>(i);
  }
}

int main() { test_matmul(8, 14); }

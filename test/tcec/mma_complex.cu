#include "utils.hpp"
#include <iostream>
#include <random>
#include <wmma_extension/tcec/complex.hpp>

template <class T, class ErrorCorrection>
constexpr double error_threshold = 0.0;
template <>
constexpr double error_threshold<half, mtk::wmma::tcec::with_ec> = 1e-5;
template <>
constexpr double
    error_threshold<nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec> =
        1e-5;
template <>
constexpr double error_threshold<half, mtk::wmma::tcec::without_ec> = 1e-2;
template <>
constexpr double error_threshold<nvcuda::wmma::precision::tf32,
                                 mtk::wmma::tcec::without_ec> = 1e-2;
template <>
constexpr double error_threshold<float, mtk::wmma::tcec::without_ec> = 4e-6;

template <unsigned N, class T, class A_Layout, class B_Layout,
          class MEM_A_Layout, class MEM_B_Layout, class Policy>
__global__ void
mma_kernel_abcd(cuComplex *const d_ptr, const cuComplex *const a_ptr,
                const cuComplex *const b_ptr, const cuComplex *const c_ptr,
                const nvcuda::wmma::layout_t cd_layout) {
  constexpr unsigned LD = N;
  __shared__ cuComplex smem[N * LD];
  mtk::test_utils::fill_zero(smem, N * LD);

  mtk::wmma::tcec::fragment_complex<nvcuda::wmma::matrix_a, N, N, N, T,
                                    A_Layout, Policy>
      frag_a;
  mtk::wmma::tcec::fragment_complex<nvcuda::wmma::matrix_b, N, N, N, T,
                                    B_Layout, Policy>
      frag_b;
  mtk::wmma::tcec::fragment_complex<nvcuda::wmma::accumulator, N, N, N, T, void,
                                    Policy>
      frag_c, frag_d;
  // Load A
  mtk::test_utils::copy_matrix(smem, LD, a_ptr, N, N, N);
  mtk::wmma::tcec::load_matrix_sync<MEM_A_Layout>(frag_a, smem, LD);

  // Load B
  mtk::test_utils::copy_matrix(smem, LD, b_ptr, N, N, N);
  mtk::wmma::tcec::load_matrix_sync<MEM_B_Layout>(frag_b, smem, LD);

  // Load C
  mtk::test_utils::copy_matrix(smem, LD, c_ptr, N, N, N);
  mtk::wmma::tcec::load_matrix_sync(frag_c, smem, LD, cd_layout);

  // Fill D
  mtk::wmma::tcec::fill_fragment(frag_d, 0.0f);

  // mma
  mtk::wmma::tcec::mma_sync(frag_d, frag_a, frag_b, frag_c);

  // Store D
  mtk::wmma::tcec::store_matrix_sync(smem, frag_d, LD, cd_layout);
  mtk::test_utils::copy_matrix(d_ptr, N, smem, LD, N, N);

  // Test for fill_zero
  mtk::wmma::tcec::fill_zero(frag_d);
}

template <unsigned N, class T, class A_Layout, class B_Layout,
          class MEM_A_Layout, class MEM_B_Layout, class Policy>
__global__ void mma_kernel_abd(cuComplex *const d_ptr,
                               const cuComplex *const a_ptr,
                               const cuComplex *const b_ptr,
                               const nvcuda::wmma::layout_t c_layout) {
  constexpr unsigned LD = N;
  __shared__ cuComplex smem[N * LD];
  mtk::test_utils::fill_zero(smem, N * LD);

  mtk::wmma::tcec::fragment_complex<nvcuda::wmma::matrix_a, N, N, N, T,
                                    A_Layout, Policy>
      frag_a;
  mtk::wmma::tcec::fragment_complex<nvcuda::wmma::matrix_b, N, N, N, T,
                                    B_Layout, Policy>
      frag_b;
  mtk::wmma::tcec::fragment_complex<nvcuda::wmma::accumulator, N, N, N, T, void,
                                    Policy>
      frag_d;
  // Load A
  mtk::test_utils::copy_matrix(smem, LD, a_ptr, N, N, N);
  mtk::wmma::tcec::load_matrix_sync<MEM_A_Layout>(frag_a, smem, LD);

  // Load B
  mtk::test_utils::copy_matrix(smem, LD, b_ptr, N, N, N);
  mtk::wmma::tcec::load_matrix_sync<MEM_B_Layout>(frag_b, smem, LD);

  // mma
  mtk::wmma::tcec::mma_sync(frag_d, frag_a, frag_b);

  // Store D
  mtk::wmma::tcec::store_matrix_sync(smem, frag_d, LD, c_layout);
  mtk::test_utils::copy_matrix(d_ptr, N, smem, LD, N, N);
}

template <unsigned N, class T, class A_Layout, class B_Layout,
          class MEM_A_Layout, class MEM_B_Layout, class Policy, bool AddC>
unsigned test_mma(const nvcuda::wmma::layout_t cd_layout) {
  cuComplex *hA, *hB, *hC, *hD;
  cudaMallocHost(&hA, N * N * sizeof(cuComplex));
  cudaMallocHost(&hB, N * N * sizeof(cuComplex));
  cudaMallocHost(&hC, N * N * sizeof(cuComplex));
  cudaMallocHost(&hD, N * N * sizeof(cuComplex));

  std::mt19937 mt(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (unsigned i = 0; i < N * N; i++) {
    hA[i].x = dist(mt);
    hB[i].x = dist(mt);
    hC[i].x = dist(mt);
    hA[i].y = dist(mt);
    hB[i].y = dist(mt);
    hC[i].y = dist(mt);
  }
  cudaDeviceSynchronize();

  if (AddC)
    mma_kernel_abcd<N, T, A_Layout, B_Layout, MEM_A_Layout, MEM_B_Layout,
                    Policy>
        <<<1, mtk::test_utils::warp_size>>>(hD, hA, hB, hC, cd_layout);
  else
    mma_kernel_abd<N, T, A_Layout, B_Layout, MEM_A_Layout, MEM_B_Layout, Policy>
        <<<1, mtk::test_utils::warp_size>>>(hD, hA, hB, cd_layout);

  const auto stat = cudaDeviceSynchronize();
  if (stat != cudaSuccess) {
    std::printf("[error] %s\n", cudaGetErrorString(stat));
  }

  double max_error = 0.;
  for (unsigned m = 0; m < N; m++) {
    for (unsigned n = 0; n < N; n++) {
      double cor_d_x = 0.;
      double cor_d_y = 0.;
      for (unsigned k = 0; k < N; k++) {
        const auto a_mem_index =
            std::is_same<MEM_A_Layout, nvcuda::wmma::col_major>::value
                ? (k * N + m)
                : (m * N + k);
        const auto b_mem_index =
            std::is_same<MEM_B_Layout, nvcuda::wmma::col_major>::value
                ? (k + n * N)
                : (n + k * N);
        const auto ca = hA[a_mem_index];
        const double cax = ca.x;
        const double cay = ca.y;
        const auto cb = hB[b_mem_index];
        const double cbx = cb.x;
        const double cby = cb.y;
        cor_d_x += (cax * cbx - cay * cby);
        cor_d_y += (cax * cby + cay * cbx);
      }
      const auto c_mem_index = (cd_layout == nvcuda::wmma::mem_col_major)
                                   ? (m + n * N)
                                   : (n + m * N);
      if (AddC) {
        cor_d_x += hC[c_mem_index].x;
        cor_d_y += hC[c_mem_index].y;
      }

      const auto diff_x = cor_d_x - hD[c_mem_index].x;
      const auto diff_y = cor_d_y - hD[c_mem_index].y;

      max_error =
          std::max(max_error, std::max(std::abs(diff_x), std::abs(diff_y)));
    }
  }

  std::printf(
      "[Type:%5s, N:%3u, A_Layout:%10s, B_Layout:%10s, MEM_A_Layout:%10s, "
      "MEM_B_Layout:%10s, C_Layout:%10s, Policy<%7s,%9s,%2u,%2u,%2u>, "
      "AddC:%3s] max_error: %e (%6s)\n",
      mtk::test_utils::to_string<T>().c_str(), N,
      mtk::test_utils::to_string<A_Layout>().c_str(),
      mtk::test_utils::to_string<B_Layout>().c_str(),
      mtk::test_utils::to_string<MEM_A_Layout>().c_str(),
      mtk::test_utils::to_string<MEM_B_Layout>().c_str(),
      (cd_layout == nvcuda::wmma::mem_col_major)
          ? mtk::test_utils::to_string<nvcuda::wmma::col_major>().c_str()
          : mtk::test_utils::to_string<nvcuda::wmma::row_major>().c_str(),
      mtk::test_utils::to_string<typename Policy::op>().c_str(),
      std::is_same<typename Policy::error_correction,
                   mtk::wmma::tcec::with_ec>::value
          ? "{w/ ec}"
          : "{w/o ec}",
      Policy::m, Policy::n, Policy::k, (AddC ? "Yes" : "No"), max_error,
      (max_error < error_threshold<T, typename Policy::error_correction>
           ? "PASSED"
           : "FAILED"));
  std::fflush(stdout);

  cudaFreeHost(hA);
  cudaFreeHost(hB);
  cudaFreeHost(hC);
  cudaFreeHost(hD);

  return max_error < error_threshold<T, typename Policy::error_correction> ? 1
                                                                           : 0;
}

int main() {
  unsigned num_passed = 0, num_tests = 0;
  // wmma FP16 test
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);

  // mma FP16 test
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          half, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);

  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::col_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::with_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed +=
      test_mma<32, half, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
               nvcuda::wmma::row_major, nvcuda::wmma::row_major,
               mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma,
                                       mtk::wmma::tcec::without_ec, 16, 8, 8>,
               false>(nvcuda::wmma::mem_row_major);
#ifdef TEST_SIMT
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type,
      false>(nvcuda::wmma::mem_row_major);
#endif
#ifdef TEST_TF32
  // wmma TF32 test
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::row_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_wmma>::type,
      false>(nvcuda::wmma::mem_row_major);

  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::col_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::col_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      true>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_col_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::with_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
  num_tests++;
  num_passed += test_mma<
      32, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major,
      nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::row_major,
      typename mtk::wmma::tcec::default_policy<nvcuda::wmma::precision::tf32,
                                               mtk::wmma::tcec::without_ec,
                                               mtk::wmma::tcec::op_mma>::type,
      false>(nvcuda::wmma::mem_row_major);
#endif
  std::printf("passed: %u / %u\n", num_passed, num_tests);
}

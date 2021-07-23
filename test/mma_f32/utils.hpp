#ifndef __HMMA_F32_F32_TEST_UTILS_HPP__
#define __HMMA_F32_F32_TEST_UTILS_HPP__
#include <cuda_fp16.h>
#include <string>
#include <wmma_extension/mma_f32/tcec.hpp>

namespace mtk {
namespace test_utils {

template <class T>
std::string to_string();
template <> std::string to_string<nvcuda::wmma::accumulator>    (){return "acc";}
template <> std::string to_string<nvcuda::wmma::matrix_a>       (){return "matrix_a";}
template <> std::string to_string<nvcuda::wmma::matrix_b>       (){return "matrix_b";}
template <> std::string to_string<nvcuda::wmma::col_major>      (){return "col_major";}
template <> std::string to_string<nvcuda::wmma::row_major>      (){return "row_major";}
template <> std::string to_string<float>                        (){return "float";}
template <> std::string to_string<half>                         (){return "half";}
template <> std::string to_string<nvcuda::wmma::precision::tf32>(){return "tf32";}
template <> std::string to_string<mtk::wmma::mma_f32::op_wmma  >(){return "op_wmma";}
template <> std::string to_string<mtk::wmma::mma_f32::op_mma   >(){return "op_mma";}
#ifdef TEST_SIMT
template <> std::string to_string<mtk::wmma::mma_f32::op_simt  >(){return "op_simt";}
#endif


constexpr unsigned warp_size = 32;

__device__ void copy_matrix(
		float* const dst, const unsigned ldd,
		const float* const src, const unsigned lds,
		const unsigned m, const unsigned n) {
	for (unsigned i = 0; i < m * n; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= m * n) return;
		const auto mm = j % m;
		const auto mn = j / m;
		dst[mm + mn * ldd] = src[mm + mn * lds];
	}
}

__device__ void fill_zero(float* const dst, const unsigned size) {
	for (unsigned i = 0; i < size; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= size) return;
		dst[j] = 0.0f;
	}
}
}
}
#endif

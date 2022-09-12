#ifndef __HMMA_F32_F32_TEST_UTILS_HPP__
#define __HMMA_F32_F32_TEST_UTILS_HPP__
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <wmma_extension/tcec/tcec.hpp>

#ifndef WMMAE_CUDA_CHECK_ERROR
#define WMMAE_CUDA_CHECK_ERROR(status) cuda_check_error(status, __FILE__, __LINE__, __func__)
#endif
#ifndef WMMAE_CUDA_CHECK_ERROR_M
#define WMMAE_CUDA_CHECK_ERROR_M(status, message) cuda_check_error(status, __FILE__, __LINE__, __func__, message)
#endif

inline void cuda_check_error(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss<< cudaGetErrorString( error );
		if(message.length() != 0){
			ss<<" : "<<message;
		}
	    ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}

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
template <> std::string to_string<mtk::wmma::tcec::op_wmma  >(){return "op_wmma";}
template <> std::string to_string<mtk::wmma::tcec::op_mma   >(){return "op_mma";}
#ifdef TEST_SIMT
template <> std::string to_string<mtk::wmma::tcec::op_simt  >(){return "op_simt";}
#endif


constexpr unsigned warp_size = 32;

template <class T>
__device__ void copy_matrix(
		T* const dst, const unsigned ldd,
		const T* const src, const unsigned lds,
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

__device__ void fill_zero(cuComplex* const dst, const unsigned size) {
	for (unsigned i = 0; i < size; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= size) return;
		dst[j].x = 0.0f;
		dst[j].y = 0.0f;
	}
}
}
}
#endif

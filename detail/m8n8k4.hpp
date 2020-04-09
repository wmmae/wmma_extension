#ifndef __M8N8K4_HPP__
#define __M8N8K4_HPP__

#include <mma.h>

namespace mtk {
namespace wmma {

template <typename T, int size> 
struct __align__(4) __frag_base {
	T x[size];
	enum {num_elements = size};
};

template <class Use, int m, int n, int k, class T, class Layout = void>
class fragment;

template <> class fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 8, 8, 4, float> : public __frag_base<float, 8>{};
template <> class fragment<nvcuda::wmma::accumulator, 8, 8, 4, half> : public __frag_base<half, 8>{};


template <class T, int size>
__device__ inline void fill_fragment(__frag_base<T, size>& f, const T v) {
#pragma unroll
	for (int i=0; i < f.num_elements; i++)
		f.x[i] = v; 
}

} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

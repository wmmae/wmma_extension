#ifndef __WMMAE_M8N8K4_HPP__
#define __WMMAE_M8N8K4_HPP__
#include <mma.h>
#include "detail/utils.hpp"

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

template <class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major>& f, const T* const p, const unsigned ldm) {
	const unsigned lane_id = mtk::detail::utils::get_lane_id();
	const unsigned col = lane_id & 0x3;
	const unsigned row_offset = ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm + row_offset;

	f.x[0] = mtk::detail::utils::cast<half>(p[mem_offset + 0]);
	f.x[1] = mtk::detail::utils::cast<half>(p[mem_offset + 1]);
	f.x[2] = mtk::detail::utils::cast<half>(p[mem_offset + 2]);
	f.x[3] = mtk::detail::utils::cast<half>(p[mem_offset + 3]);
}

template <class T>
__device__ inline void load_matrix_sync(fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major>& f, const T* const p, const unsigned ldm) {
	const unsigned lane_id = mtk::detail::utils::get_lane_id();
	const unsigned col = (lane_id & 0x3) + ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm;

	f.x[0] = mtk::detail::utils::cast<half>(p[mem_offset]);
	f.x[1] = mtk::detail::utils::cast<half>(p[mem_offset + ldm]);
	f.x[2] = mtk::detail::utils::cast<half>(p[mem_offset + ldm * 2]);
	f.x[3] = mtk::detail::utils::cast<half>(p[mem_offset + ldm * 3]);
}

__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major>& a, fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& c) {
	asm(R"({
	mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32
	{%0, %1, %2, %3, %4, %5, %6, %7},
	{%8, %9},
	{%10, %11},
	{%12, %13, %14, %15, %16, %17, %18, %19};
})"
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)),
			  "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)),
			  "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
			);
}



} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

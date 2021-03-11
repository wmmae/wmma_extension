#ifndef __WMMAE_M8N8K4_HPP__
#define __WMMAE_M8N8K4_HPP__
#include <mma.h>
#include "common.hpp"

namespace mtk {
namespace wmma {

template <> class fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 8, 8, 4, float> : public __frag_base<float, 8>{};
template <> class fragment<nvcuda::wmma::accumulator, 8, 8, 4, half> : public __frag_base<half, 8>{};


template <class T, int size>
__device__ inline void fill_fragment(__frag_base<T, size>& f, const T v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = v; 
}

template <class T, class Func>
__device__ inline void foreach(mtk::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned col = lane_id & 0x3;
	const unsigned row_offset = ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm + row_offset;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class T, class Func>
__device__ inline void foreach(mtk::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0x3) + ((lane_id >> 4) << 2);
	const unsigned mem_offset = row * ldm;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class T, class Func>
__device__ inline void foreach(mtk::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned col = (lane_id & 0x3) + ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class T, class Func>
__device__ inline void foreach(mtk::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = lane_id & 0x3;
	const unsigned col_offset = ((lane_id >> 4) << 2);
	const unsigned mem_offset = row * ldm + col_offset;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

#define WMMAE_MMA884_F32_F32(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};}" \
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])); \
}

WMMAE_MMA884_F32_F32(col, col);
WMMAE_MMA884_F32_F32(row, col);
WMMAE_MMA884_F32_F32(col, row);
WMMAE_MMA884_F32_F32(row, row);

#define WMMAE_MMA884_F16_F32(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f16.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15};}" \
			: "=r"(*reinterpret_cast<unsigned*>(d.x + 0)), "=r"(*reinterpret_cast<unsigned*>(d.x + 2)), "=r"(*reinterpret_cast<unsigned*>(d.x + 4)), "=r"(*reinterpret_cast<unsigned*>(d.x + 6)) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])); \
}

WMMAE_MMA884_F16_F32(col, col);
WMMAE_MMA884_F16_F32(row, col);
WMMAE_MMA884_F16_F32(col, row);
WMMAE_MMA884_F16_F32(row, row);

#define WMMAE_MMA884_F32_F16(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f32.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%12, %13, %14, %15};}" \
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 0)), "r"(*reinterpret_cast<const unsigned*>(c.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 4)), "r"(*reinterpret_cast<const unsigned*>(c.x + 6))); \
}

WMMAE_MMA884_F32_F16(col, col);
WMMAE_MMA884_F32_F16(row, col);
WMMAE_MMA884_F32_F16(col, row);
WMMAE_MMA884_F32_F16(row, row);

#define WMMAE_MMA884_F16_F16(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};}" \
			: "=r"(*reinterpret_cast<unsigned*>(d.x + 0)), "=r"(*reinterpret_cast<unsigned*>(d.x + 2)), "=r"(*reinterpret_cast<unsigned*>(d.x + 4)), "=r"(*reinterpret_cast<unsigned*>(d.x + 6)) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 0)), "r"(*reinterpret_cast<const unsigned*>(c.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 4)), "r"(*reinterpret_cast<const unsigned*>(c.x + 6))); \
}

WMMAE_MMA884_F16_F16(col, col);
WMMAE_MMA884_F16_F16(row, col);
WMMAE_MMA884_F16_F16(col, row);
WMMAE_MMA884_F16_F16(row, row);

// Debug function
template <class MatrixType, int M, int N, int K, class MemMajor, class T>
__device__ inline void print_fragment(const mtk::wmma::fragment<MatrixType, M, N, K, T, MemMajor>& frag, const char* name = "") {
	if ((threadIdx.x & 0x1f) == 0) {
		if (name[0] != '\0') {
			printf("%s = \n", name);
		}
	}
	for (unsigned i = 0; i < warpSize; i++) {
		if (i == (threadIdx.x & 0x1f)) {
			for (unsigned j = 0; j < frag.num_elements; j++) {
				const auto v = mtk::wmma::detail::common::cast<float>(frag.x[j]);
				if (v == 0.0f) {
					printf(" %.3e ", 0.0f);
				} else if (v > 0) {
					printf(" %.3e ", v);
				} else {
					printf("%.3e ", v);
				}
			}
			printf("\n");
		}
		__syncthreads();
	}
}

} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

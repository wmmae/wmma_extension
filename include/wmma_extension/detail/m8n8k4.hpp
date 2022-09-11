#ifndef __WMMAE_M8N8K4_HPP__
#define __WMMAE_M8N8K4_HPP__
#include <mma.h>
#include "common.hpp"

namespace mtk {
namespace wmma {
namespace mma {
template <> class fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major> : public __frag_base<half, 4>{};
template <> class fragment<nvcuda::wmma::accumulator, 8, 8, 4, float> : public __frag_base<float, 8>{};
template <> class fragment<nvcuda::wmma::accumulator, 8, 8, 4, half> : public __frag_base<half, 8>{};


template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	constexpr unsigned ldm = 8;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned col = lane_id & 0x3;
	const unsigned row_offset = ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm + row_offset;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	constexpr unsigned ldm = 4;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0x3) + ((lane_id >> 4) << 2);
	const unsigned mem_offset = row * ldm;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	constexpr unsigned ldm = 4;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned col = (lane_id & 0x3) + ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	constexpr unsigned ldm = 8;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = lane_id & 0x3;
	const unsigned col_offset = ((lane_id >> 4) << 2);
	const unsigned mem_offset = row * ldm + col_offset;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, half, void>& f, const nvcuda::wmma::layout_t layout, Func func) {
	constexpr unsigned ldm = 8;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0x3) + ((lane_id & 0x10) >> 2);
	if (layout == nvcuda::wmma::mem_col_major) {
#pragma unroll
		for (unsigned i = 0; i < 8; i++)
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, row + i * ldm);}
	} else {
		const unsigned index_offset = row * ldm;
#pragma unroll
		for (unsigned i = 0; i < 8; i++)
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, index_offset + i);}
	}
}

template <class Func>
__device__ inline void foreach(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, float, void>& f, const nvcuda::wmma::layout_t layout, Func func) {
	constexpr unsigned ldm = 8;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row_offset = (lane_id & 0x1) + ((lane_id & 0x10) >> 2);
	const unsigned col_offset = (lane_id & 0x2);

	if (layout == nvcuda::wmma::mem_col_major) {
#pragma unroll
		for (unsigned i = 0; i < f.num_elements; i++) {
			const unsigned row = row_offset + (i & 0x2);
			const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, row + col * ldm);}
		}
	} else {
#pragma unroll
		for (unsigned i = 0; i < f.num_elements; i++) {
			const unsigned row = row_offset + (i & 0x2);
			const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, row * ldm + col);}
		}
	}
}

// foreach_ij
template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned col = lane_id & 0x3;
	const unsigned row_offset = ((lane_id >> 4) << 2);

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, row_offset + 0, col);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, row_offset + 1, col);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, row_offset + 2, col);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, row_offset + 3, col);}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0x3) + ((lane_id >> 4) << 2);

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, row, 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, row, 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, row, 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, row, 3);}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned col = (lane_id & 0x3) + ((lane_id >> 4) << 2);

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, 0, col);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, 1, col);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, 2, col);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, 3, col);}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = lane_id & 0x3;
	const unsigned col_offset = ((lane_id >> 4) << 2);

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, row, col_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, row, col_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, row, col_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, row, col_offset + 3);}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, half, void>& f, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0x3) + ((lane_id & 0x10) >> 2);
#pragma unroll
	for (unsigned i = 0; i < 8; i++) {
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, row, i);}
	}
}

template <class Func>
__device__ inline void foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, float, void>& f, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row_offset = (lane_id & 0x1) + ((lane_id & 0x10) >> 2);
	const unsigned col_offset = (lane_id & 0x2);

#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++) {
		const unsigned row = row_offset + (i & 0x2);
		const unsigned col = col_offset + ((i & 0x1) + (i & 0x4));
		{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, row, col);}
	}
}

// foreach_v
template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	constexpr unsigned ldm = 16;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (lane_id % 4) return;
	const unsigned col = lane_id & 0x3;
	const unsigned row_offset = ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm + row_offset;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (lane_id & 0b10010) return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, ((lane_id & 0x1) << 2) + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, ((lane_id & 0x1) << 2) + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, ((lane_id & 0x1) << 2) + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, ((lane_id & 0x1) << 2) + 3);}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::col_major>& f, Func func) {
	constexpr unsigned ldm = 16;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (lane_id & 0b10011) return;
	const unsigned col = (lane_id & 0x3) + ((lane_id >> 4) << 2);
	const unsigned mem_offset = col * ldm;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, mem_offset + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, mem_offset + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, mem_offset + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, mem_offset + 3);}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::row_major>& f, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	if (lane_id & 0b11) return;

	{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, ((lane_id >> 4) << 2) + 0);}
	{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, ((lane_id >> 4) << 2) + 1);}
	{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, ((lane_id >> 4) << 2) + 2);}
	{const unsigned frag_index_list[1] = {3};func(frag_index_list, 1, ((lane_id >> 4) << 2) + 3);}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, half, void>& f, const nvcuda::wmma::layout_t layout, Func func) {
	constexpr unsigned ldm = 8;
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();
	const unsigned row = (lane_id & 0x3) + ((lane_id & 0x10) >> 2);
	if (layout == nvcuda::wmma::mem_col_major) {
		{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, row);}
	} else {
		if (lane_id & 0b10011) return;
		const unsigned index_offset = row * ldm;
#pragma unroll
		for (unsigned i = 0; i < 8; i++)
			{const unsigned frag_index_list[1] = {i};func(frag_index_list, 1, index_offset + i);}
	}
}

template <class Func>
__device__ inline void foreach_v(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, float, void>& f, const nvcuda::wmma::layout_t layout, Func func) {
	const unsigned lane_id = mtk::wmma::detail::common::get_lane_id();

	if (layout == nvcuda::wmma::mem_col_major) {
		if (lane_id & 0b10) return;
		{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, ((lane_id >> 4) << 2) + (lane_id & 0x1) + 0);}
		{const unsigned frag_index_list[1] = {2};func(frag_index_list, 1, ((lane_id >> 4) << 2) + (lane_id & 0x1) + 2);}
	} else {
		if (lane_id & 0b10001) return;
		{const unsigned frag_index_list[1] = {0};func(frag_index_list, 1, (lane_id & 0x2) + 0);}
		{const unsigned frag_index_list[1] = {1};func(frag_index_list, 1, (lane_id & 0x2) + 1);}
		{const unsigned frag_index_list[1] = {4};func(frag_index_list, 1, (lane_id & 0x2) + 4);}
		{const unsigned frag_index_list[1] = {5};func(frag_index_list, 1, (lane_id & 0x2) + 5);}
	}
}

#define WMMAE_MMA884_F32_F32(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, const fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};}" \
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])); \
}

WMMAE_MMA884_F32_F32(col, col);
WMMAE_MMA884_F32_F32(row, col);
WMMAE_MMA884_F32_F32(col, row);
WMMAE_MMA884_F32_F32(row, row);

#define WMMAE_MMA884_F16_F32(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, const fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f16.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15};}" \
			: "=r"(*reinterpret_cast<unsigned*>(d.x + 0)), "=r"(*reinterpret_cast<unsigned*>(d.x + 2)), "=r"(*reinterpret_cast<unsigned*>(d.x + 4)), "=r"(*reinterpret_cast<unsigned*>(d.x + 6)) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]), "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])); \
}

WMMAE_MMA884_F16_F32(col, col);
WMMAE_MMA884_F16_F32(row, col);
WMMAE_MMA884_F16_F32(col, row);
WMMAE_MMA884_F16_F32(row, row);

#define WMMAE_MMA884_F32_F16(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, float>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, const fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f32.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%12, %13, %14, %15};}" \
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7]) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 0)), "r"(*reinterpret_cast<const unsigned*>(c.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 4)), "r"(*reinterpret_cast<const unsigned*>(c.x + 6))); \
}

WMMAE_MMA884_F32_F16(col, col);
WMMAE_MMA884_F32_F16(row, col);
WMMAE_MMA884_F32_F16(col, row);
WMMAE_MMA884_F32_F16(row, row);

#define WMMAE_MMA884_F16_F16(A_LAYOUT, B_LAYOUT) \
__device__ inline void mma_sync(fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& d, const fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, nvcuda::wmma::A_LAYOUT##_major>& a, const fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, nvcuda::wmma::B_LAYOUT##_major>& b, const fragment<nvcuda::wmma::accumulator, 8, 8, 4, half>& c) { \
	asm("{mma.sync.aligned.m8n8k4."#A_LAYOUT"."#B_LAYOUT".f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};}" \
			: "=r"(*reinterpret_cast<unsigned*>(d.x + 0)), "=r"(*reinterpret_cast<unsigned*>(d.x + 2)), "=r"(*reinterpret_cast<unsigned*>(d.x + 4)), "=r"(*reinterpret_cast<unsigned*>(d.x + 6)) \
			: "r"(*reinterpret_cast<const unsigned*>(a.x)), "r"(*reinterpret_cast<const unsigned*>(a.x + 2)), "r"(*reinterpret_cast<const unsigned*>(b.x)), "r"(*reinterpret_cast<const unsigned*>(b.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 0)), "r"(*reinterpret_cast<const unsigned*>(c.x + 2)), "r"(*reinterpret_cast<const unsigned*>(c.x + 4)), "r"(*reinterpret_cast<const unsigned*>(c.x + 6))); \
}

WMMAE_MMA884_F16_F16(col, col);
WMMAE_MMA884_F16_F16(row, col);
WMMAE_MMA884_F16_F16(col, row);
WMMAE_MMA884_F16_F16(row, row);
} // namespace mma
} // namespace wmma
} // namespace mtk

#endif /* end of include guard */

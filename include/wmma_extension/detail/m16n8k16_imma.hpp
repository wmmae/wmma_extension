#ifndef __WMMAE_M16N8K16_IMMA_HPP__
#define __WMMAE_M16N8K16_IMMA_HPP__
#include <mma.h>
#include "common.hpp"

namespace mtk {
namespace wmma {
namespace mma {
template <> class fragment<nvcuda::wmma::matrix_a   , 16, 8, 16, std::int8_t , nvcuda::wmma::row_major> : public __frag_base<std::int32_t , 2>{};
template <> class fragment<nvcuda::wmma::matrix_b   , 16, 8, 16, std::int8_t , nvcuda::wmma::col_major> : public __frag_base<std::int32_t , 1>{};
template <> class fragment<nvcuda::wmma::matrix_a   , 16, 8, 16, std::uint8_t, nvcuda::wmma::row_major> : public __frag_base<std::uint32_t, 2>{};
template <> class fragment<nvcuda::wmma::matrix_b   , 16, 8, 16, std::uint8_t, nvcuda::wmma::col_major> : public __frag_base<std::uint32_t, 1>{};
template <> class fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t> : public __frag_base<std::int32_t, 4>{};

// foreach

__device__ inline void mma_sync(
		fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t>& d,
		const fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::int8_t, nvcuda::wmma::row_major>& a,
		const fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::int8_t, nvcuda::wmma::col_major>& b,
		const fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t>& c) {
	asm(R"({
    mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
      {%0, %1, %2, %3},
      {%4, %5},
      {%6, %7},
      {%8, %9, %10, %11};
})"
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
			: "r"(*reinterpret_cast<const unsigned*>(a.x)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 2)),
			"r"(*reinterpret_cast<const unsigned*>(b.x)),
			"f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

__device__ inline void mma_sync(
		fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t>& d,
		const fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::uint8_t, nvcuda::wmma::row_major>& a,
		const fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::uint8_t, nvcuda::wmma::col_major>& b,
		const fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t>& c) {
	asm(R"({
    mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32
      {%0, %1, %2, %3},
      {%4, %5},
      {%6, %7},
      {%8, %9, %10, %11};
})"
			: "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
			: "r"(*reinterpret_cast<const unsigned*>(a.x)),
			"r"(*reinterpret_cast<const unsigned*>(a.x + 2)),
			"r"(*reinterpret_cast<const unsigned*>(b.x)),
			"f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}
} // namespace mma
} // namespace wmma
} // namespace mtk

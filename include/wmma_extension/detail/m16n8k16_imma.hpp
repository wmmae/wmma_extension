#ifndef __WMMAE_M16N8K16_IMMA_HPP__
#define __WMMAE_M16N8K16_IMMA_HPP__
#include "common.hpp"
#include <mma.h>

namespace mtk {
namespace wmma {
namespace mma {
template <>
class fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::int8_t,
               nvcuda::wmma::row_major> : public __frag_base<std::int8_t, 8> {};
template <>
class fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::int8_t,
               nvcuda::wmma::col_major> : public __frag_base<std::int8_t, 4> {};
template <>
class fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::uint8_t,
               nvcuda::wmma::row_major> : public __frag_base<std::uint8_t, 8> {
};
template <>
class fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::uint8_t,
               nvcuda::wmma::col_major> : public __frag_base<std::uint8_t, 4> {
};
template <>
class fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t>
    : public __frag_base<std::int32_t, 4> {};

// foreach
template <class Func>
__device__ inline void foreach (
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::int8_t,
                             nvcuda::wmma::row_major> &frag,
    Func func) {
  const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 4;
  const unsigned row = mtk::wmma::detail::common::get_lane_id() / 4;
  for (unsigned i = 0; i < 4; i++) {
    {
      const unsigned frag_index_list[1] = {i + 0};
      func(frag_index_list, 1, col + i + (row + 0) * 16);
    }
    {
      const unsigned frag_index_list[1] = {i + 4};
      func(frag_index_list, 1, col + i + (row + 8) * 16);
    }
  }
}

template <class Func>
__device__ inline void foreach (
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::int8_t,
                             nvcuda::wmma::col_major> &frag,
    Func func) {
  const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
  const unsigned row = (mtk::wmma::detail::common::get_lane_id() % 4) * 4;
  for (unsigned i = 0; i < 4; i++) {
    {
      const unsigned frag_index_list[1] = {i};
      func(frag_index_list, 1, row + i + col * 16);
    }
  }
}

template <class Func>
__device__ inline void foreach (
    mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t>
        &frag,
    const nvcuda::wmma::layout_t layout, Func func) {
  const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;
  const unsigned row_offset = mtk::wmma::detail::common::get_lane_id() / 4;

  for (unsigned i = 0; i < 2; i++) {
    const auto row = row_offset + i * 8;
    if (layout == nvcuda::wmma::mem_col_major) {
      {
        const unsigned frag_index_list[1] = {(i * 2 + 0)};
        func(frag_index_list, 1, row + (col + 0) * 16);
      }
      {
        const unsigned frag_index_list[1] = {(i * 2 + 1)};
        func(frag_index_list, 1, row + (col + 1) * 16);
      }
    } else {
      {
        const unsigned frag_index_list[1] = {(i * 2 + 0)};
        func(frag_index_list, 1, row * 8 + (col + 0));
      }
      {
        const unsigned frag_index_list[1] = {(i * 2 + 1)};
        func(frag_index_list, 1, row * 8 + (col + 1));
      }
    }
  }
}

// foreach_ij
template <class Func>
__device__ inline void
foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16,
                                    std::int8_t, nvcuda::wmma::row_major> &frag,
           Func func) {
  const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 4;
  const unsigned row = mtk::wmma::detail::common::get_lane_id() / 4;

  for (unsigned i = 0; i < 4; i++) {
    {
      const unsigned frag_index_list[1] = {i + 0};
      func(frag_index_list, 1, (row + 0), col + i);
    }
    {
      const unsigned frag_index_list[1] = {i + 4};
      func(frag_index_list, 1, (row + 8), col + i);
    }
  }
}

template <class Func>
__device__ inline void
foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16,
                                    std::int8_t, nvcuda::wmma::col_major> &frag,
           Func func) {
  const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
  const unsigned row = (mtk::wmma::detail::common::get_lane_id() % 4) * 4;

  for (unsigned i = 0; i < 4; i++) {
    {
      const unsigned frag_index_list[1] = {i};
      func(frag_index_list, 1, row + i, col);
    }
  }
}

template <class Func>
__device__ inline void
foreach_ij(mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16,
                                    std::int32_t> &frag,
           const nvcuda::wmma::layout_t layout, Func func) {
  const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 2;
  const unsigned row_offset = mtk::wmma::detail::common::get_lane_id() / 4;

  for (unsigned i = 0; i < 2; i++) {
    const auto row = row_offset + i * 8;
    if (layout == nvcuda::wmma::mem_col_major) {
      {
        const unsigned frag_index_list[1] = {(i * 2 + 0)};
        func(frag_index_list, 1, row, col + 0);
      }
      {
        const unsigned frag_index_list[1] = {(i * 2 + 1)};
        func(frag_index_list, 1, row, col + 1);
      }
    } else {
      {
        const unsigned frag_index_list[1] = {(i * 2 + 0)};
        func(frag_index_list, 1, row, col + 0);
      }
      {
        const unsigned frag_index_list[1] = {(i * 2 + 1)};
        func(frag_index_list, 1, row, col + 1);
      }
    }
  }
}

// load_matrix_sync
__device__ inline void load_matrix_sync(
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::int8_t,
                             nvcuda::wmma::row_major> &frag,
    const std::int8_t *const ptr, const unsigned ldm, const bool sync = true) {
  const unsigned col = (mtk::wmma::detail::common::get_lane_id() % 4) * 4;
  const unsigned row = mtk::wmma::detail::common::get_lane_id() / 4;

  *reinterpret_cast<std::int32_t *>(&frag.x[0]) =
      *reinterpret_cast<const std::int32_t *>(&ptr[(row + 0) * ldm + col]);
  *reinterpret_cast<std::int32_t *>(&frag.x[4]) =
      *reinterpret_cast<const std::int32_t *>(&ptr[(row + 8) * ldm + col]);

  if (sync)
    __syncwarp();
}

__device__ inline void load_matrix_sync(
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::int8_t,
                             nvcuda::wmma::col_major> &frag,
    const std::int8_t *const ptr, const unsigned ldm, const bool sync = true) {
  const unsigned col = mtk::wmma::detail::common::get_lane_id() / 4;
  const unsigned row = (mtk::wmma::detail::common::get_lane_id() % 4) * 4;

  *reinterpret_cast<std::int32_t *>(&frag.x[0]) =
      *reinterpret_cast<const std::int32_t *>(&ptr[col * ldm + row]);

  if (sync)
    __syncwarp();
}

// alias
template <class Func>
__device__ inline void foreach (
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::uint8_t,
                             nvcuda::wmma::row_major> &frag,
    Func func) {
  foreach (
      *reinterpret_cast<
          mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16,
                                   std::int8_t, nvcuda::wmma::row_major> *>(
          &frag),
      func)
    ;
}

template <class Func>
__device__ inline void foreach (
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::uint8_t,
                             nvcuda::wmma::col_major> &frag,
    Func func) {
  foreach (
      *reinterpret_cast<
          mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16,
                                   std::int8_t, nvcuda::wmma::col_major> *>(
          &frag),
      func)
    ;
}

__device__ inline void load_matrix_sync(
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::uint8_t,
                             nvcuda::wmma::row_major> &frag,
    const std::uint8_t *const ptr, const unsigned ldm, const bool sync = true) {
  load_matrix_sync(
      *reinterpret_cast<
          mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a, 16, 8, 16,
                                   std::int8_t, nvcuda::wmma::row_major> *>(
          &frag),
      reinterpret_cast<const std::int8_t *>(ptr), ldm, sync);
}

__device__ inline void load_matrix_sync(
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::uint8_t,
                             nvcuda::wmma::col_major> &frag,
    const std::uint8_t *const ptr, const unsigned ldm, const bool sync = true) {
  load_matrix_sync(
      *reinterpret_cast<
          mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b, 16, 8, 16,
                                   std::int8_t, nvcuda::wmma::col_major> *>(
          &frag),
      reinterpret_cast<const std::int8_t *>(ptr), ldm, sync);
}

__device__ inline void mma_sync(
    fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t> &d,
    const fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::int8_t,
                   nvcuda::wmma::row_major> &a,
    const fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::int8_t,
                   nvcuda::wmma::col_major> &b,
    const fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t> &c) {
  asm(R"({
    mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
      {%0, %1, %2, %3},
      {%4, %5},
      {%6},
      {%7, %8, %9, %10};
})"
      : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
      : "r"(*reinterpret_cast<const unsigned *>(a.x)),
        "r"(*reinterpret_cast<const unsigned *>(a.x + 4)),
        "r"(*reinterpret_cast<const unsigned *>(b.x)), "r"(c.x[0]), "r"(c.x[1]),
        "r"(c.x[2]), "r"(c.x[3]));
}

__device__ inline void mma_sync(
    fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t> &d,
    const fragment<nvcuda::wmma::matrix_a, 16, 8, 16, std::uint8_t,
                   nvcuda::wmma::row_major> &a,
    const fragment<nvcuda::wmma::matrix_b, 16, 8, 16, std::uint8_t,
                   nvcuda::wmma::col_major> &b,
    const fragment<nvcuda::wmma::accumulator, 16, 8, 16, std::int32_t> &c) {
  asm(R"({
    mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32
      {%0, %1, %2, %3},
      {%4, %5},
      {%6},
      {%7, %8, %9, %10};
})"
      : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
      : "r"(*reinterpret_cast<const unsigned *>(a.x)),
        "r"(*reinterpret_cast<const unsigned *>(a.x + 4)),
        "r"(*reinterpret_cast<const unsigned *>(b.x)), "r"(c.x[0]), "r"(c.x[1]),
        "r"(c.x[2]), "r"(c.x[3]));
}
} // namespace mma
} // namespace wmma
} // namespace mtk
#endif

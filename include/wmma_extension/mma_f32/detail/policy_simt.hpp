#ifndef __WMMAE_HMMA_F32_F32_DETAIL_POLICY_SIMT_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_POLICY_SIMT_HPP__
#include <mma.h>
#include "policy.hpp"
#include "wmma_extension_simt_include.hpp"

namespace mtk {
namespace wmma {

namespace mma_f32 {

// Instruction policy
struct op_simt;

namespace detail {
// ===================================
// Default policy selector
// ===================================
template <class T>
struct default_policy<T                            , mtk::wmma::mma_f32::without_ec, mtk::wmma::mma_f32::op_simt> {
	using type = mtk::wmma::mma_f32::Policy<mtk::wmma::mma_f32::op_simt, mtk::wmma::mma_f32::without_ec, 16, 16, 16>;
};

// ===================================
// Default fragment selector
// ===================================
template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct default_fragment<Use, T, Layout, Policy<op_simt, ErrorCorrection, fm, fn, fk>> {
	using type = mtk::wmma::mma_simt::fragment<Use, fm, fn, fk, T, Layout>;
};
} // namespace detail

} // namespace mma_f32
} // namespace wmma
} // namespace mtk
#endif

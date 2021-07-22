#ifndef __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#include <mma.h>
#include "wmma_extension_include.hpp"

namespace mtk {
namespace wmma {

namespace mma_f32 {

// Instruction policy
struct op_mma;
struct op_wmma;

// Error correction policy
struct op_with_error_correction;
struct op_without_error_correction;

template <class Op, class ErrorCorrection, int m_, int n_, int k_>
struct Policy {
	using op = Op;
	using error_correction = ErrorCorrection;
	static const int m = m_;
	static const int n = n_;
	static const int k = k_;
};

namespace detail {
// ===================================
// Default policy selector
// ===================================
template <class T, class ErrorCorrection = mtk::wmma::mma_f32::op_with_error_correction, class Op = mtk::wmma::mma_f32::op_wmma>
struct default_policy;
template <class ErrorCorrection>
struct default_policy<half                         , ErrorCorrection, mtk::wmma::mma_f32::op_wmma> {using type = mtk::wmma::mma_f32::Policy<mtk::wmma::mma_f32::op_wmma, ErrorCorrection, 16, 16, 16>;};
template <class ErrorCorrection>
struct default_policy<nvcuda::wmma::precision::tf32, ErrorCorrection, mtk::wmma::mma_f32::op_wmma> {using type = mtk::wmma::mma_f32::Policy<mtk::wmma::mma_f32::op_wmma, ErrorCorrection, 16, 16, 8 >;};
template <class ErrorCorrection>
struct default_policy<half                         , ErrorCorrection, mtk::wmma::mma_f32::op_mma > {using type = mtk::wmma::mma_f32::Policy<mtk::wmma::mma_f32::op_mma , ErrorCorrection, 16, 8 , 16>;};


// ===================================
// Default fragment selector
// ===================================
template <class Use, class T, class Layout, class Policy>
struct default_fragment;

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct default_fragment<Use, T, Layout, Policy<op_wmma, ErrorCorrection, fm, fn, fk>> {
	using type = nvcuda::wmma::fragment<Use, fm, fn, fk, T, Layout>;
};

template <class Use, class T, class Layout, class ErrorCorrection, int fm, int fn, int fk>
struct default_fragment<Use, T, Layout, Policy<op_mma , ErrorCorrection, fm, fn, fk>> {
	using type = mtk::wmma::mma::fragment<Use, fm, fn, fk, T, Layout>;
};
} // namespace detail

} // namespace mma_f32
} // namespace wmma
} // namespace mtk
#endif

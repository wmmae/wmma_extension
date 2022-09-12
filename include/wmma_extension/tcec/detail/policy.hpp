#ifndef __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#define __WMMAE_HMMA_F32_F32_DETAIL_POLICY_HPP__
#include <mma.h>
#include "wmma_extension_include.hpp"

namespace mtk {
namespace wmma {

namespace tcec {

// Instruction policy
struct op_mma;
struct op_wmma;

// Error correction policy
struct with_ec;
struct without_ec;
// Alias for compatibility
using op_with_error_correction = with_ec;
using op_without_error_correction = without_ec;

struct sm_70;
struct sm_75;
struct sm_80;
struct sm_86;
struct sm_not_specified;

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
template <class T, class ErrorCorrection = mtk::wmma::tcec::with_ec, class Op = mtk::wmma::tcec::op_wmma, class Sm = mtk::wmma::tcec::sm_not_specified>
struct default_policy;

template <class ErrorCorrection, class Sm>
struct default_policy<half                         , ErrorCorrection, mtk::wmma::tcec::op_wmma, Sm>
{using type = mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, 16, 16, 16>;};

template <class ErrorCorrection, class Sm>
struct default_policy<nvcuda::wmma::precision::tf32, ErrorCorrection, mtk::wmma::tcec::op_wmma, Sm>
{using type = mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_wmma, ErrorCorrection, 16, 16, 8 >;};

template <class ErrorCorrection, class Sm>
struct default_policy<half                         , ErrorCorrection, mtk::wmma::tcec::op_mma , Sm>
{using type = mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma , ErrorCorrection, 16, 8 , 16>;};

template <class ErrorCorrection>
struct default_policy<half                         , ErrorCorrection, mtk::wmma::tcec::op_mma , mtk::wmma::tcec::sm_75>
{using type = mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma , ErrorCorrection, 16, 8 , 8>;};

template <class ErrorCorrection, class Sm>
struct default_policy<nvcuda::wmma::precision::tf32, ErrorCorrection, mtk::wmma::tcec::op_mma , Sm>
{using type = mtk::wmma::tcec::Policy<mtk::wmma::tcec::op_mma , ErrorCorrection, 16, 8 , 8 >;};


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

template <class T, class ErrorCorrection = mtk::wmma::tcec::with_ec, class Op = mtk::wmma::tcec::op_wmma>
using default_policy = detail::default_policy<T, ErrorCorrection, Op>;

} // namespace tcec
} // namespace wmma
} // namespace mtk
#endif

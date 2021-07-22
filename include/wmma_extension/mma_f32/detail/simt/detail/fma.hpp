#ifndef __WMMAE_MMA_SIMT_DETAIL_FMA_HPP__
#define __WMMAE_MMA_SIMT_DETAIL_FMA_HPP__
#include "common.hpp"

namespace mtk {
namespace wmma {
namespace mma_simt {
namespace detail {

template <class A_T, class B_T, class C_T>
struct fma {
	virtual __device__ C_T operator() (const A_T a, const B_T b, const C_T c) const {
		const auto fa = cast<float>(a);
		const auto fb = cast<float>(b);
		const auto fc = cast<float>(c);
		return fa * fb + fc;
	}
};

template <>
struct fma<double, double, double> {
	virtual __device__ double operator() (const double a, const double b, const double c) const {
		return a * b + c;
	}
};
} // namespace detail
} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif

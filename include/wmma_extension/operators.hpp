#ifndef __WMMAE_OPERATORS__
#define __WMMAE_OPERATORS__
#include "detail/operators.hpp"
#include "detail/common.hpp"

template <class Use, int M, int N, int K, class Type, class Layout>
__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator+(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	return mtk::wmma::ops::add<Use, M, N, K, Type, Layout>{}(a, b);
}

template <class Use, int M, int N, int K, class Type, class Layout>
__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator-(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	return mtk::wmma::ops::sub<Use, M, N, K, Type, Layout>{}(a, b);
}

template <class Use, int M, int N, int K, class Type, class Layout>
__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator*(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const typename nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>::storage_element_type b) {
	return mtk::wmma::ops::mul<Use, M, N, K, Type, Layout>{}(a, b);
}

template <class Use, int M, int N, int K, class Type, class Layout>
__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator/(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const typename nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>::storage_element_type b) {
	return mtk::wmma::ops::div<Use, M, N, K, Type, Layout>{}(a, b);
}

namespace mtk {
namespace wmma {
template <class Use, int M, int N, int K, class Type, class Layout>
__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> fma(
		const typename nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>::storage_element_type alpha,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	return mtk::wmma::ops::fma<Use, M, N, K, Type, Layout>{}(alpha, a, b);
}

template <class Use, int M, int N, int K, class Type, class Layout>
__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> fma(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const typename nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>::storage_element_type alpha,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	return mtk::wmma::ops::fma<Use, M, N, K, Type, Layout>{}(alpha, a, b);
}
} // namespace wmma
} // namespace mtk
#endif

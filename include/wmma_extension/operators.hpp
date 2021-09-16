#ifndef __WMMAE_OPERATORS__
#define __WMMAE_OPERATORS__
#include "detail/operators.hpp"
#include "detail/common.hpp"

template <class Use, int M, int N, int K, class Type, class Layout>
nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator+(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	return mtk::wmma::detail::add{}(a, b);
}

template <class Use, int M, int N, int K, class Type, class Layout>
nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator-(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	return mtk::wmma::detail::sub{}(a, b);
}
#endif

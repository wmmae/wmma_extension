#ifndef __WMMAE_OPERATORS__
#define __WMMAE_OPERATORS__
#include <mma.h>

template <class Use, int M, int N, int K, class Type, class Layout>
nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator+(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& res;
	for (unsigned i = 0; i < res.num_elements; i++) {
		res.x[i] = a.x[i] + b.x[i];
	}
	return res;
}

template <class Use, int M, int N, int K, class Type, class Layout>
nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator-(
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
		const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& res;
	for (unsigned i = 0; i < res.num_elements; i++) {
		res.x[i] = a.x[i] - b.x[i];
	}
	return res;
}
#endif

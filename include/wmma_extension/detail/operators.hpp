#ifndef __WMMAE_DETAIL_OPERATORS__
#define __WMMAE_DETAIL_OPERATORS__
#include <mma.h>

namespace mtk {
namespace wmma {
namespace ops {
template <class Use, int M, int N, int K, class Type, class Layout>
struct add {
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = a.x[i] + b.x[i];
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Layout>
struct add<Use, M, N, K, half, Layout> {
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& res;
		for (unsigned i = 0; i < res.num_elements / 2; i++) {
			reinterpret_cast<half2>(res.x)[i] = __hadd2(reinterpret_cast<half2>(a.x)[i], reinterpret_cast<half2>(b.x)[i]);
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Type, class Layout>
struct sub {
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = a.x[i] + b.x[i];
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Layout>
struct sub<Use, M, N, K, half, Layout> {
	nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& res;
		for (unsigned i = 0; i < res.num_elements / 2; i++) {
			reinterpret_cast<half2>(res.x)[i] = __hsub2(reinterpret_cast<half2>(a.x)[i], reinterpret_cast<half2>(b.x)[i]);
		}
		return res;
	}
};
} // namespace ops
} // namespace wmma
} // namespace mtk
#endif

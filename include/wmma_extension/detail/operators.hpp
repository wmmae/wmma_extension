#ifndef __WMMAE_DETAIL_OPERATORS__
#define __WMMAE_DETAIL_OPERATORS__
#include <mma.h>

namespace mtk {
namespace wmma {
namespace ops {

// Add
template <class Use, int M, int N, int K, class Type, class Layout>
struct add {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = a.x[i] + b.x[i];
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Layout>
struct add<Use, M, N, K, half, Layout> {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, half, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, half, Layout> res;
		for (unsigned i = 0; i < res.num_elements / 2; i++) {
			reinterpret_cast<half2*>(res.x)[i] = __hadd2(reinterpret_cast<const half2*>(a.x)[i], reinterpret_cast<const half2*>(b.x)[i]);
		}
		return res;
	}
};

// Sub
template <class Use, int M, int N, int K, class Type, class Layout>
struct sub {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = a.x[i] - b.x[i];
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Layout>
struct sub<Use, M, N, K, half, Layout> {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, half, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, half, Layout> res;
		for (unsigned i = 0; i < res.num_elements / 2; i++) {
			reinterpret_cast<half2*>(res.x)[i] = __hsub2(reinterpret_cast<const half2*>(a.x)[i], reinterpret_cast<const half2*>(b.x)[i]);
		}
		return res;
	}
};

// Mul
template <class Use, int M, int N, int K, class Type, class Layout>
struct mul {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const typename nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>::storage_element_type b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = a.x[i] * b;
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Layout>
struct mul<Use, M, N, K, half, Layout> {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, half, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& a,
			const typename nvcuda::wmma::fragment<Use, M, N, K, half, Layout>::storage_element_type b) {
		nvcuda::wmma::fragment<Use, M, N, K, half, Layout> res;
		for (unsigned i = 0; i < res.num_elements / 2; i++) {
			reinterpret_cast<half2*>(res.x)[i] = __hmul2(reinterpret_cast<const half2*>(a.x)[i], __half2half2(b));
		}
		return res;
	}
};

// Fma
template <class Use, int M, int N, int K, class Type, class Layout>
struct fma {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const Type alpha,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = __fmaf_rn(alpha, a.x[i], b.x[i]);
		}
		return res;
	}
};

template <class Use, int M, int N, int K, class Layout>
struct fma<Use, M, N, K, half, Layout> {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, half, Layout> operator()(
			const half alpha,
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& a,
			const nvcuda::wmma::fragment<Use, M, N, K, half, Layout>& b) {
		nvcuda::wmma::fragment<Use, M, N, K, half, Layout> res;
		for (unsigned i = 0; i < res.num_elements / 2; i++) {
			reinterpret_cast<half2*>(res.x)[i] = __hfma2(__half2half2(alpha),
					reinterpret_cast<const half2*>(a.x)[i],
					reinterpret_cast<const half2*>(b.x)[i]);
		}
		return res;
	}
};

// Div
template <class Use, int M, int N, int K, class Type, class Layout>
struct div {
	__device__ nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> operator()(
			const nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>& a,
			const typename nvcuda::wmma::fragment<Use, M, N, K, Type, Layout>::storage_element_type b) {
		nvcuda::wmma::fragment<Use, M, N, K, Type, Layout> res;
		for (unsigned i = 0; i < res.num_elements; i++) {
			res.x[i] = a.x[i] / b;
		}
		return res;
	}
};
} // namespace ops
} // namespace wmma
} // namespace mtk
#endif

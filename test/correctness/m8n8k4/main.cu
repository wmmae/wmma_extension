#include <wmma_extension.hpp>

template <class T, class S, class a_layout, class b_layout>
__global__ void m8n8k4_test(T* const d, const half* const a, const half* const b, const S* const c) {
	mtk::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, half, a_layout> frag_a;
	mtk::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, half, b_layout> frag_b;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, T> frag_c;
	mtk::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, S> frag_d;

	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_d);
}

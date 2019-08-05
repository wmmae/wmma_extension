# WMMA API

## Required
- CUDA (9.2 or later)
- C++ (11 or later)

## Sample
```cuda
__global__ void kernel() {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_b;

	__shared__ float vec16[16];

	mtk::wmma::load_vector_sync(frag_a, vec16);
	mtk::wmma::load_vector_sync(frag_b, vec16);

	mtk::wmma::make_identity_matrix(frag_c);
}
```

## Implemented functions
### load_vector_sync
- Arguments
  - dst_fragment : Destination fragment (matrix_a / matrix_b, (16, 16, 16), half / float, col_major / row_major)
  - src_pointer  : Source pointer (No alignment restriction)

### make_identity_matrix
- Arguments
  - dst_fragment : Destination fragment (accumulator, (16, 16, 16), half / float)

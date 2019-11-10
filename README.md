# WMMA API Extension
This extension provides a function for
- loading vector as a fragment
- making identity matrix fragment

without using shared memory.

## Required
- CUDA (9.2 or later)
- C++ (11 or later)

## Sample
```cuda
#include <mma.h>
#incldue <wmma_extension.hpp>

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
![load_matrix](https://gitlab.momo86.net/mutsuki/wmma-extension/raw/master/docs/load_matrix.svg)
- Arguments
  - dst_fragment : Destination fragment (matrix_a / matrix_b, (16, 16, 16), half / float, col_major / row_major)
  - src_pointer  : Source pointer (No alignment restriction)

### load_matrix_with_operation_sync
```cuda
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	__shared__ compute_t vec16[16 * 16];
	mtk::wmma::load_matrix_with_operation_sync(
			frag,
			ptr,
			[](const unsigned index, const compute_t value) {return static_cast<half>(value * 2.0f);}
		);
```
- Arguments
  - dst_fragment : Destination fragment (matrix_a / matrix_b, (16, 16, 16), half / float, col_major / row_major)
  - src_pointer  : Source pointer (No alignment restriction)
  - func         : Element-wise function. Return type must be `half`.

### make_identity_matrix
- Arguments
  - dst_fragment : Destination fragment (accumulator, (16, 16, 16), half / float)

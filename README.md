<img src='./docs/wmmae.svg' width=200>

# WMMA API Extension

This extension provides a function for
- loading vector as a fragment
- storing fragment as a vector
- making eye matrix fragment
- making a fragment of a matrix with element-wise operations
- etc

without using extra shared memory.

This extension also provides a C++ interface of experimental function:
- `mma.m8n8k4` for `f16/f32` (sm_70)

which is available in only PTX.
See [detail](./docs/m8n8k4.md).

## Required
- CUDA (9.2 or later)
- C++ (11 or later)

## Supported fragment
- sm_70: (16, 16, 16)
- sm_75: (16, 16, 16)
- sm_80: (16, 16, 16)

## Sample
```cuda
#include <mma.h>
#include <wmma_extension.hpp>

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
![load_matrix](docs/load_vector-en.svg)
- Arguments
  - dst_fragment : Destination fragment (`nvcuda::wmma::matrix_a` / `nvcuda::wmma::matrix_b`, (16, 16, 16), `half`, `nvcuda::wmma::col_major` / `nvcuda::wmma::row_major`)
  - src_pointer  : Source pointer (No alignment restriction)

### store_vector_sync
![store_matrix](docs/store_vector-en.svg)
- Arguments
  - dst_pointer  : Destination pointer (No alignment restriction)
  - src_fragment : Source fragment (`nvcuda::wmma::accumulator` , (16, 16, 16), `half` / `float`)
  - layout       : `nvcuda::wmma::mem_col_major` / `nvcuda::wmma::mem_row_major`

### load_matrix_with_operation
This function is used for making a fragment of a matrix with element-wise operations.
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
__shared__ compute_t matrix[16 * 16];
mtk::wmma::load_matrix_with_operation(
		frag,
		matrix,
		[](const unsigned index, const compute_t value) -> half {return static_cast<half>(value * 2.0f);}
	);
```

- Arguments
  - dst_fragment : Destination fragment (`matrix_a` / `matrix_b`, (16, 16, 16), `half` / `float`, `col_major` / `row_major`)
  - src_pointer  : Source pointer (No alignment restriction)
  - func         : Element-wise function. Return type has to be `half`.

The first argument of `func` is an index of `fragment.x[]` and the second one is a value of `fragment.x[]` if `func` is an identity function.

### foreach
This function calculates the mapping of memory and fragmrnt.
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
__shared__ compute_t matrix[16 * 16];
mtk::wmma::foreach(
		frag,
		[&](const unsigned frag_index, const unsigned mem_index) {frag_b.x[frag_index] = matrix[mem_index];}
	);
```

- Arguments
  - frag         : fragment. This argument is used for template argument deduction.
  - func         : like a function which sets fragments from `fragment_index` and `mem_index`.

### make_direct_product_fragments
This function is used for computing direct product of two vectors (u and v) with accuracy correction.

![make_direct_product_fragments](docs/make_direct_product_fragments-en.svg)

- Arguments
  - frag_a : Destination fragment (`matrix_a` and `col_major` / `matrix_b` and `row_major`, (16, 16, 16), `half`)
  - x      : x (`float` / `half`)
  - dx     : diff vector of `x` (`x` - toFloat(toHalf(`x`))) (`float` / `half`)

### make_eye
![load_matrix](docs/make_eye-en.svg)
- Arguments
  - dst_fragment : Destination fragment (`accumulator`, (16, 16, 16), `half` / `float`)
  - alpha : diagonal element

### make_identity_matrix
This function is equivalent to `make_eye(frag, 1.0f)`
- Arguments
  - dst_fragment : Destination fragment (`accumulator`, (16, 16, 16), `float` / `half`)

### fill_zero
- Argument
  - dst_fragment : Destination fragment (`float` / `half`)

### print_fragment
This function output the elements of a fragment.
- Arguments
  - frag : Target fragment
  - name : printing name of fragment (`char*`)

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

**Important!!**

Tensor Core does not have backward compatibility.
Please specify an appropriate virtual architecture for real GPU when you use this library.
For instance, a program which is compiled with `-arch=sm_70` does not work correctly on Ampare GPUs.

## Required
- CUDA (9.2 or later)
- C++ (11 or later)

## Supported fragment
- sm_70: ((16, 16, 16), fp16/fp32)
- sm_75: ((16, 16, 16), fp16/fp32)
- sm_80: ((16, 16, 16), fp16/fp32), ((16, 16, 8), tf32)

## Sample
```cuda
#include <mma.h>
#include <wmma_extension/wmma_extension.hpp>

__global__ void kernel() {
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_b;

	__shared__ float vec16[16];

	mtk::wmma::load_vector(frag_a, vec16);
	mtk::wmma::load_vector(frag_b, vec16);

	mtk::wmma::make_identity_matrix(frag_c);
}
```

## Implemented functions
### Primitive functions
#### foreach
This function calculates the mapping of memory and fragment elements.
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
__shared__ compute_t matrix[16 * 16];
mtk::wmma::foreach<decltype(frag_b)>(
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			const auto m = mem_index % 16;
			const auto n = mem_index / 16;
			for (unsigned i = 0; i < fragment_index_count; i++)
				frag_b.x[frag_index_list[i]] = convert_to<half>(matrix[n * 16 + m]);
		});
	);
```

- Arguments
  - func         : a function which sets fragments from `fragment_index_list`, `fragmnt_index_count` and `mem_index`.

#### foreach_v
##### For matrix A/B
This function calculates the mapping of a given vector and fragment elements.
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
__shared__ compute_t vector[16];
mtk::wmma::foreach_v<decltype(frag_b)>(
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				frag_b.x[frag_index_list[i]] = convert_to<half>(vector[mem_index]);
		});
// is equivalent to `load_vector`
```

- Arguments
  - func         : a function which sets fragments from `fragment_index_list`, `fragmnt_index_count` and `mem_index`.

##### For accumulator
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;
__shared__ compute_t vector[16];
mtk::wmma::foreach_v<decltype(frag_c)>(nvcuda::wmma::mem_col_major,
		[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
			for (unsigned i = 0; i < fragment_index_count; i++)
				vector[mem_index] = convert_to<compute_t>(frag_c.x[frag_index_list[i]]);
		});
// is equivalent to `store_vector`
```

### Other functions
#### load_vector
![load_matrix](docs/load_vector-en.svg)
- Arguments
  - dst_fragment : Destination fragment (`nvcuda::wmma::matrix_a` / `nvcuda::wmma::matrix_b`, `nvcuda::wmma::col_major` / `nvcuda::wmma::row_major`)
  - src_pointer  : Source pointer (No alignment restriction)

#### store_vector
![store_matrix](docs/store_vector-en.svg)
- Arguments
  - dst_pointer  : Destination pointer (No alignment restriction)
  - src_fragment : Source fragment (`nvcuda::wmma::accumulator` , `half` / `float`)
  - layout       : `nvcuda::wmma::mem_col_major` / `nvcuda::wmma::mem_row_major`

#### load_matrix_with_operation
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
  - dst_fragment : Destination fragment (`matrix_a` / `matrix_b`)
  - src_pointer  : Source pointer (No alignment restriction)
  - func         : Element-wise function. Return type has to be `half`.

The first argument of `func` is an index of `fragment.x[]` and the second one is a value of `fragment.x[]` if `func` is an identity function.

#### load_vector_with_rounding
This function is used for making a fragment of a vector with explicitly converting to `tf32`.
TF32 TensorCore performs RZ rounding when we use WMMA API but it is not good for accuracy.

- Arguments
  - dst_fragment : Destination fragment (`nvcuda::wmma::matrix_a` / `nvcuda::wmma::matrix_b`, (16, 16, 8), `tf32`, `nvcuda::wmma::col_major` / `nvcuda::wmma::row_major`)
  - src_pointer  : Source pointer (No alignment restriction)

#### make_direct_product_fragments (A)
This function is used for computing direct product of two vectors (u and v) with accuracy correction.

![make_direct_product_fragments](docs/make_direct_product_fragments-en.svg)

- Arguments
  - frag_a/b : Destination fragment (`matrix_a` needs to be `col_major` / `matrix_b` needs to be  `row_major`)
  - x        : x (`float` / `half`)
  - dx       : diff vector of `x` (`x` - toFloat(toHalf(`x`))) (`float` / `half`)

- Detail
`frag_a` x `frag_b` conmutes a direct product `u` x `v` with error correction.

#### make_direct_product_fragments (B)

Its computation is same with `make_direct_product_fragments (A)` but arguments are different.

- Arguments
  - frag_a/b : Destination fragment (`matrix_a` needs to be `col_major` / `matrix_b` needs to be  `row_major`)
  - x        : x (`float` / `half`)

dx is automatically computed in this method.

#### make_eye
![load_matrix](docs/make_eye-en.svg)
- Arguments
  - dst_fragment : Destination fragment (`accumulator`)
  - alpha : diagonal element

#### make_identity_matrix
This function is equivalent to `make_eye(frag, 1.0f)`
- Arguments
  - dst_fragment : Destination fragment (`accumulator`)

#### fill_zero
- Argument
  - dst_fragment : Destination fragment

### Debug functions

#### print_fragment
This function output the elements of a fragment.
- Arguments
  - frag : Target fragment
  - name : printing name of fragment (`char*`, optional)

<img src='./docs/wmmae.svg' width=200>

# WMMA API Extension

This extension provides features for
- mapping between memory and fragment
- operationf for vectors
    - loading a vector as a fragment
    - storing a fragment as a vector
- making eye matrix fragment
- etc

without using extra shared memory.

This extension also provides a C++ interface of experimental function:
- `mma.m8n8k4` for `f16/f32` (sm_70)

which is available in only PTX.
See [detail](./docs/m8n8k4.md).

**Caution!!**

WMMA API does not have backward compatibility.
Please specify an appropriate virtual architecture for real GPU when you use this library.
For instance, a program which is compiled with `-arch=sm_70` does not work correctly on Ampere GPUs.

## Requirements
- CUDA (9.2 or later)
- C++ (17 or later)

## Supported fragment
- sm_70: ((16, 16, 16), fp16/fp32)
- sm_75: ((16, 16, 16), fp16/fp32)
- sm_80: ((16, 16, 16), fp16/fp32), ((16, 16, 8), tf32)

# Functions
## Primitive functions
### foreach
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

### foreach_v
#### For matrix A/B
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

#### For accumulator
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


## Functions for vector
## Sample
```cuda
#include <mma.h>
#include <wmma_extension/wmma_extension.hpp>

__global__ void kernel() {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;

    __shared__ float vec16[16];

    mtk::wmma::load_vector(frag_a, vec16);
    mtk::wmma::load_vector(frag_b, vec16);

    nvcuda::wmma::fill_fragment(frag_c, 0.0f);
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    mtk::wmma::store_vector(vec16, frag_c, nvcuda::wmma::mem_col_major);
}
```

## Other functions
### make_identity_matrix / add_eye
![load_matrix](docs/make_eye-en.svg)
- Arguments
  - dst_fragment : Destination fragment (`accumulator`)
  - alpha : diagonal element

### fill_zero
- Argument
  - dst_fragment : Destination fragment

### Debug functions

#### print_fragment
This function output the elements of a fragment.
- Arguments
  - frag : Target fragment
  - name : printing name of fragment (`char*`, optional)

# LICENSE
MIT

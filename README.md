<img src='./docs/wmmae.svg' width=200>

# WMMA API Extension

This extension provides features for
- mapping between memory and fragment (primitive functions)
- operationf for vectors
    - loading a vector as a fragment
    - storing a fragment as a vector
- making eye matrix fragment
- C++ interface for `mma` instructions
- Error Correction (TCEC) for SGEMM emulation [[detail](./docs/mma_f32.md)]
- arithmetic operators for fragments (`+, -, *, /, fma`) [[detail](./docs/ops.md)]
- utils [[detail](./docs/utils.md)]
- etc

without using extra shared memory.

**Caution!!**

WMMA API does not have backward compatibility.
Please specify an appropriate virtual architecture for real GPU when you use this library.
For instance, a program which is compiled with `-arch=sm_70` may not work correctly on Ampere GPUs.

## Requirements
- CUDA (10.2 or later)
- C++ (17 or later)

## Supported architecures / fragment
- [x] sm_70: ((16, 16, 16), fp16/fp32)
- [x] sm_75: ((16, 16, 16), fp16/fp32)
- [x] sm_80: ((16, 16, 16), fp16/fp32), ((16, 16, 8), tf32)
- [x] sm_89: ((16, 16, 16), fp16/fp32), ((16, 16, 8), tf32)
- [ ] sm_90: * `wgmma` instruction is not supported yet

# Functions
## Primitive functions
### foreach
This function calculates the mapping of the memory and fragment elements.
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
```

### foreach_ij
This function calculates the mapping of the matrix element position (i,j) and fragment elements.
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
__shared__ compute_t matrix[16 * 16];
mtk::wmma::foreach_ij<decltype(frag_b)>(
        [&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned i, const unsigned j) {
            for (unsigned f = 0; f < fragment_index_count; f++)
                frag_b.x[frag_index_list[f]] = convert_to<half>(matrix[j * 16 + i]);
        });
```

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

### map
This function returns the mapping of matrix element (i, j) and fragment element (tid, fid)
```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
unsigned tid_list[2];
unsigned fid_list[2];
unsigned list_size;
mtk::wmma::map<decltype(frag_b)>(tid_list, fid_list, list_size, i, j);
for (unsigned k = 0; k < list_size; k++) {
  if ((threadIdx.x & 0x1f) == tid_list[k]) {
    frag_b.x[fid_list[k]] = 3.0f;
  }
}
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

## C++ interface of `mma` instructions

```cpp
#include <wmma_extension/wmma_mma.hpp>

__global__ void kernel(float* const d, const half* const a, const half* const b, const float* const c) {
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_a   , 16, 8, 16, half, nvcuda::wmma::col_major> frag_a;
    mtk::wmma::mma::fragment<nvcuda::wmma::matrix_b   , 16, 8, 16, half, nvcuda::wmma::col_major> frag_b;
    mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16, float> frag_c;
    mtk::wmma::mma::fragment<nvcuda::wmma::accumulator, 16, 8, 16, float> frag_d;

    mtk::wmma::mma::load_matrix_sync(frag_a, a, 16);
    mtk::wmma::mma::load_matrix_sync(frag_b, b, 8);
    mtk::wmma::mma::load_matrix_sync(frag_c, c, 16, nvcuda::wmma::mem_col_major);

    mtk::wmma::mma::mma_sync(frag_d, frag_a, frag_b, frag_c);

    mtk::wmma::mma::store_matrix_sync(d, frag_d, 16, nvcuda::wmma::mem_col_major);
}
```

### Supported fragments

| shape    |  type                | arch            |
|:-------- |:-------------------- |:--------------- |
| m16n8k16 | `half`               | sm_80 or higher |
| m16n8k8  | `half`               | sm_75 or higher |
| m16n8k8  | `nvcuda::wmma::tf32` | sm_80 or higher |
| m8n8k4   | `half`               | sm_70, sm_75    |

### Supported functions
- `foreach`
- `foreach_v`
- `load_matrix_sync`
- `store_matrix_sync`
- `fill_fragment`
- `fill_zero`

# LICENSE
MIT

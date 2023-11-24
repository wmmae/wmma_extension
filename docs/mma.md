# C++ interface of `mma` instructions

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

## Supported fragments

| shape    |  A,B type            |  C, D type           | arch            |
|:-------- |:-------------------- |:-------------------- |:--------------- |
| m16n8k16 | `half`               | `float` / `half`     | sm_80 or higher |
| m16n8k8  | `half`               | `float` / `half`     | sm_75 or higher |
| m16n8k8  | `nvcuda::wmma::tf32` | `float`              | sm_80 or higher |
| m8n8k4   | `half`               | `float` / `half`     | sm_70, sm_75    |
| m16n8k16 | `int8` / `uint8`     | `int32`              | sm_80 or higher |
| m16n8k32 | `int8` / `uint8`     | `int32`              | sm_80 or higher |

## Supported functions
- `foreach`
- `foreach_ij`
- `load_matrix_sync`
- `store_matrix_sync`
- `fill_fragment`
- `fill_zero`

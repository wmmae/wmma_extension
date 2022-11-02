# WMMA Extension for single precision matmul using Tensor Cores and error correction technique (TCEC)

## Correction technique
See [our paper](https://arxiv.org/abs/2203.03341).

## Requirements
- CUDA
  - CUDA >= 10.0 for HMMA-FP16
  - CUDA >= 11.1 for HMMA-TF32

- C++ >= 14

## Installation
1. Clone [wmma_extension](https://github.com/wmmae/wmma_extension)
```bash
git clone https://github.com/wmmae/wmma_extension
```

## Sample code
```cuda
// sample.cu
// nvcc -I./path/to/wmma_extension/include/ -std=c++17 sample.cu ...
//
#include <wmma_extension/tcec/tcec.hpp>

template <unsigned N>
__global__ void mma_kernel(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
    __shared__ float smem[N * N];
    fill_zero(smem, N * N);

    mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> frag_a;
    mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_b, N, N, N, half, nvcuda::wmma::col_major> frag_b;
    mtk::wmma::tcec::fragment<nvcuda::wmma::accumulator, N, N, N, half> frag_c, frag_d;

    // Load A
    // copy_matrix(smem, N, a_ptr, N, N, N);
    mtk::wmma::tcec::load_matrix_sync(frag_a, smem, N);

    // Load B
    // copy_matrix(smem, N, b_ptr, N, N, N);
    mtk::wmma::tcec::load_matrix_sync(frag_b, smem, N);

    // Load C
    // copy_matrix(smem, N, c_ptr, N, N, N);
    mtk::wmma::tcec::load_matrix_sync(frag_c, smem, N, nvcuda::wmma::mem_col_major);

    // Fill D
    mtk::wmma::tcec::fill_fragment(frag_d, 0.0f);

    // mma
    mtk::wmma::tcec::mma_sync(frag_d, frag_a, frag_b, frag_c);

    // Store D
    mtk::wmma::tcec::store_matrix_sync(smem, frag_d, N, nvcuda::wmma::mem_col_major);
    //copy_matrix(d_ptr, N, smem, N, N, N);
}
```

## Fragment
```cpp
template <class Use, int m, int n, int k, class T, class Layout = void, Policy = typename mtk::wmma::tcec::default_policy<T>::type>
struct fragment;
```

### Template arguments
`mtk::wmma::tcec::fragment` is a fragment for this computation.
It contains arrays of `nvcuda::wmma::fragment`.
- `m`, `n` and `k` have to be a multiple of `Policy::m`, `Policy::n` and `Policy::k` respectively.
You can get a default policy using `mtk::wmma::tcec::default_policy<T>::type`.
- `k` has to be a multiple of 16 when `T` is `half` and 8 when `T` is `nvcuda::wmma::precision::tf32`.
- `T` is `half` or `nvcuda::wmma::precision::tf32`. Unlike `nvcuda::wmma::fragment`, even if `Use` is `nvcuda::wmma::accumulator`, the same is true.
- `Policy` is a concept of `mtk::wmma::tcec::Policy<Op, ErrorCorrection, fm, fn, fk>`.
  - `Op` : `mtk::wmma::tcec::op_mma` / `mtk::wmma::tcec::op_wmma`
  - `ErrorCorrection` : `mtk::wmma::tcec::with_ec` / `mtk::wmma::tcec::without_ec`
  - `fm`, `fn`, `fk` is a size of internal fragments.

### Policy
`default_policy` can make `Policy` easily.
```cuda
using policy = mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec, mtk::wmma::tcec::op_mma>::type;
```

## Supported fragment

| fm | fn | fk | LayoutA | LayoutB | Type  | Operation      | Supported arch |
| -- | -- | -- | ------- | ------- | ----- | -------------- | ---------------|
| 16 | 16 | 16 | col/row | col/row | half  | Arch dependent | sm_70 or later |
| 16 | 16 | 16 | col/row | col/row | tf32  | wmma           | sm_80 or later |
| 16 | 8  | 8  | row     | col     | tf32  | mma            | sm_80 or later |
| 16 | 8  | 8  | row     | col     | half  | mma            | sm_75 or later |
| 16 | 8  | 16 | row     | col     | half  | mma            | sm_80 or later |

### Note
To get detault policy for `sm_75` and `op_mma`, specify the architecture as follows:
```cuda
using policy = mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec, mtk::wmma::tcec::op_mma, mtk::wmma::tcec::sm_75>::type;
```

### Member variables/functions
- Member variable `element_type` is `float`
- Member function `x(index)` and `dx(index)` return the referrence of a elements.

## Functions
- `mtk::wmma::tcec::fill_fragment`
- `mtk::wmma::tcec::load_matrix_sync`
- `mtk::wmma::tcec::store_matrix_sync`
- `mtk::wmma::tcec::mma_sync`

- `mtk::wmma::tcec::mma_rz_sync`
- `mtk::wmma::tcec::load_vector`
- `mtk::wmma::tcec::store_vector`
- `mtk::wmma::tcec::fill_zero`

### Note
While some `fragment` only supports either `row` or `col`, `load_matrix_sync` function can load both memory layout matrices using an additional template parameter.

```cpp
// e.g.
using policy = mtk::wmma::tcec::default_policy<half, mtk::wmma::tcec::with_ec, mtk::wmma::tcec::op_mma>::type;
mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, 32, 32, 32, half, nvcuda::wmma::row_major, policy> frag_a;

mtk::wmma::tcec::load_matrix_sync<nvcuda::wmma::col_major>(frag_a, matrix_ptr, ldm);
```


## Rounding mode
To specify the rounding mode in `+C` operation, use functions as follows.
- `mtk::wmma::tcec::mma_rn_sync`
- `mtk::wmma::tcec::mma_rz_sync`

### Default rounding mode
| op         | rounding mode |
| ---------- | ------------- |
| with_ec    | RN            |
| without_ec | RZ            |

Read [our paper](https://arxiv.org/abs/2203.03341) for detail.

## SIMT Core computation

This library provides fragments and functionf for mma operations using CUDA SIMT Core with the same API as WMMA API.

| fm | fn | fk | LayoutA | LayoutB | Type  | Operation      | Supported arch |
| -- | -- | -- | ------- | ------- | ----- | -------------- | ---------------|
| 16 | 16 | 16 | col/row | col/row | float | simt           | sm_70 or later |

### Policy
```cuda
using simt_policy = typename mtk::wmma::tcec::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type;

mtk::wmma::tcec::fragment<nvcuda::wmma::matrix_a, N, N, N, float, nvcuda::wmma::col_major, simt_policy> frag_a;
```

## Complex type
```cuda
mtk::wmma::tcec::fragment_complex<nvcuda::wmma::matrix_a, N, N, N, float, nvcuda::wmma::col_major> frag_a;
// or
using policy = typename mtk::wmma::tcec::default_policy<float, mtk::wmma::tcec::without_ec, mtk::wmma::tcec::op_simt>::type;
mtk::wmma::tcec::fragment_complex<nvcuda::wmma::matrix_a, N, N, N, float, nvcuda::wmma::col_major, policy> frag_a;
```

### Supported functions
- `mtk::wmma::tcec::fill_fragment`
- `mtk::wmma::tcec::load_matrix_sync`
- `mtk::wmma::tcec::store_matrix_sync`
- `mtk::wmma::tcec::mma_sync`

- `mtk::wmma::tcec::mma_rz_sync`
- `mtk::wmma::tcec::fill_zero`

See [test code](../test/tcec/mma_complex.cu) for more detail.

# Arithmetic operators for fragments

## Supported operators

|  op  | A type | B type | C type |
|:----:|:------:|:------:|:------:|
| `+`  | `fragment` | `fragment` ||
| `-`  | `fragment` | `fragment` ||
| `*`  | `fragment` | `fragment::storage_element_t` ||
| `/`  | `fragment` | `fragment::storage_element_t` ||
| `mtk::wmma::fma`  | `fragment` | `fragment::storage_element_t` | `fragment` |
| `mtk::wmma::fma`  | `fragment::storage_element_t` | `fragment` | `fragment` |

## Example

```cpp
#include <wmma_extension/operators.hpp>

nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;

const auto frag_a0 = frag_a0 + frag_a1 * __float2half(2.0f);
```

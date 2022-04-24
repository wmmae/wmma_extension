# Utils

## Type conversion
```cpp
const auto dst_val = mtk::wmma::utils::cast<DST_TYPE>(src_val);
```

## Asynchronous D2S data copy
```cpp
mtk::wmma::utils::cp_async::cp_async<N>(dst_ptr, src_ptr);
mtk::wmma::utils::cp_async::commit();
mtk::wmma::utils::cp_async::wait_group<i>();
mtk::wmma::utils::cp_async::wait_all();
```

- `N` is data size in byte. (4, 8, 16)

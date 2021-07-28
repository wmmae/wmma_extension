#ifndef __WMMAE_TEST_COMMON_HPP__
#define __WMMAE_TEST_COMMON_HPP__
#include <string>

namespace mtk {
namespace test_utils {
template <class T>
std::string get_string();
template <> std::string get_string<float>() {return "float";}
template <> std::string get_string<half >() {return "half";}
template <> std::string get_string<nvcuda::wmma::precision::tf32>() {return "tf32";}
template <> std::string get_string<nvcuda::wmma::col_major>() {return "col_major";}
template <> std::string get_string<nvcuda::wmma::row_major>() {return "row_major";}
template <> std::string get_string<void>() {return "row_major";}
template <> std::string get_string<nvcuda::wmma::matrix_a>()  {return "matrix_a";}
template <> std::string get_string<nvcuda::wmma::matrix_b>()  {return "matrix_b";}
template <> std::string get_string<nvcuda::wmma::accumulator>()  {return "accumulator";}
} // namespace test_utils
} // namespace mtk
#endif

#include <wmma_extension/utils.hpp>

int main() {
	const auto a = mtk::wmma::utils::cast<half >(1.0f);
	const auto b = mtk::wmma::utils::cast<float>(a);
}

#ifndef __WMMAE_TCEC_DETAIL_PRINT_HPP__
#define __WMMAE_TCEC_DETAIL_PRINT_HPP__
#include <stdio.h>
#include "policy.hpp"
#include "common.hpp"
#include "scale.hpp"

namespace mtk {
namespace wmma {
namespace tcec {
template <class Use, int m, int n, int k, class Type, class Layout, class OP, int fm, int fn, int fk>
__device__ void print_fragment(
		const mtk::wmma::tcec::fragment<Use, m, n, k, Type, Layout, Policy<OP, mtk::wmma::tcec::with_ec, fm, fn, fk>>& frag,
		const char* const name = "") {
	if (*name != '\0' && mtk::wmma::detail::common::get_lane_id() == 0) {
		printf("%s = \n", name);
	}

	__syncwarp();
	for (unsigned i = 0; i < 32; i++) {
		if (i == mtk::wmma::detail::common::get_lane_id()) {
			for (unsigned i = 0; i < frag.num_elements; i++) {
				printf("(%+.3e)+(%+.3e) ", frag.x(i), detail::correction_scale_1<Type>(frag.dx(i)));
			}
			printf("\n");
		}
		__syncwarp();
	}
}
template <class Use, int m, int n, int k, class Type, class Layout, class OP, int fm, int fn, int fk>
__device__ void print_fragment(
		const mtk::wmma::tcec::fragment<Use, m, n, k, Type, Layout, Policy<OP, mtk::wmma::tcec::without_ec, fm, fn, fk>>& frag,
		const char* const name = "") {
	if (*name != '\0' && mtk::wmma::detail::common::get_lane_id() == 0) {
		printf("%s = \n", name);
	}

	__syncwarp();
	for (unsigned i = 0; i < 32; i++) {
		if (i == mtk::wmma::detail::common::get_lane_id()) {
			for (unsigned i = 0; i < frag.num_elements; i++) {
				printf("%+.3e ", frag.x(i));
			}
			printf("\n");
		}
		__syncwarp();
	}
}
} // namespace mtk
} // namespace wmma
} // namespace tcec
#endif

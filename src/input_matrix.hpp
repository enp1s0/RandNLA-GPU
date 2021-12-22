#ifndef __RSVD_TEST_INPUT_MATRIX_HPP__
#define __RSVD_TEST_INPUT_MATRIX_HPP__
#include <string>

namespace mtk {
namespace rsvd_test {
void get_input_matrix(
		float* const ptr,
		const std::string input_matrix_name,
		const std::size_t m,
		const std::size_t n,
		const std::uint64_t seed
		);
} // namespace rsvd_test
} // namespace mtk
#endif

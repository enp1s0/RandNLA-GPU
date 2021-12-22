#include <vector>
#include <input_matrix.hpp>
#include <lapacke.h>

void mtk::rsvd_test::gen_latms_matrix(
		float* const ptr,
		const std::size_t ld,
		const std::size_t m,
		const std::size_t n,
		const std::size_t rank,
		const std::uint64_t seed
		) {
	int iseed[4] = {0, 1, 2, static_cast<int>(seed)};

	const auto s_vec_len = std::min(m, n);
	std::vector<float> s_vec(s_vec_len);

	std::size_t i;
	for (i = 0; i < rank; i++) {
		s_vec[i] = (s_vec_len - i) / static_cast<float>(s_vec_len);
	}
	for (; i < s_vec_len; i++) {
		s_vec[i] = 0.f;
	}

	LAPACKE_slatms(
		LAPACK_COL_MAJOR,
		m, n,
		'S',
		iseed,
		'N',
		s_vec.data(),
		0,
		0,
		0,
		m - 1,
		n - 1,
		'N',
		ptr,
		ld
		);
}

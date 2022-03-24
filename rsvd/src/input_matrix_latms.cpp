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

void mtk::rsvd_test::gen_latms_sigmoid_matrix(
		float* const ptr,
		const std::size_t ld,
		const std::size_t m,
		const std::size_t n,
		const std::size_t p,
		const std::uint64_t seed
		) {
	int iseed[4] = {0, 1, 2, static_cast<int>(seed)};

	const auto s_vec_len = std::min(m, n);
	std::vector<float> s_vec(s_vec_len);

	std::size_t i;
	for (i = 0; i < std::min(m, n); i++) {
		s_vec[i] = 1 - 1. / (1. + std::exp(-(i - p)));
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

void mtk::rsvd_test::gen_latms_designed_matrix(
		float* const ptr,
		const std::size_t ld,
		const std::size_t m,
		const std::size_t n,
		const std::size_t p,
		const int log10_s_p,
		const std::string name,
		const std::uint64_t seed
		) {
	int iseed[4] = {0, 1, 2, static_cast<int>(seed)};

	const auto s_vec_len = std::min(m, n);
	std::vector<float> s_vec(s_vec_len);

	const auto target_s = std::pow<float>(10., log10_s_p);

	if (name == "linear") {
		std::size_t i;
		for (i = 0; i < p; i++) {
			s_vec[i] = - (1 - target_s) / p * i + 1;
		}
		for (; i < std::min(m, n); i++) {
			s_vec[i] = target_s;
		}
	} else if (name == "exp") {
		const auto q = std::log2(target_s) / p;
		for (std::size_t i = 0; i < std::min(m, n); i++) {
			s_vec[i] = std::pow<float>(2.f, q * i);
		}
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

#include <vector>
#include <input_matrix.hpp>
#include <lapacke.h>
#include <random>
#include <memory>

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

void mtk::rsvd_test::gen_cauchy_matrix(
		float* const ptr,
		const std::size_t ld,
		const std::size_t m,
		const std::size_t n,
		const std::uint64_t seed
		) {
	std::mt19937 mt(seed);
	std::uniform_real_distribution<float> dist(-0.001, 0.001);
	auto x_array = std::unique_ptr<float>(new float[n]);
	auto y_array = std::unique_ptr<float>(new float[m]);

	for (std::size_t i = 0; i < n; i++) {
		x_array.get()[i] = dist(mt);
	}
	for (std::size_t i = 0; i < m; i++) {
		y_array.get()[i] = dist(mt);
	}

#pragma omp parallel for collapse(2)
	for (std::size_t x = 0; x < n; x++) {
		for (std::size_t y = 0; y < m; y++) {
			const auto v = std::abs(x_array.get()[x] - y_array.get()[y]) + 0.0001;
			ptr[y + x * ld] = 1.f / v;
		}
	}
}

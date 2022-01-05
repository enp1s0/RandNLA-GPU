#include <input_matrix.hpp>
#include <matfile/matfile.hpp>
#include <chrono>

int mtk::rsvd_test::exist_input_matrix(
	const std::string input_matrix_name,
	const std::size_t m, const std::size_t n,
	const std::uint64_t seed
	) {
	const std::string mat_file_name = input_matrix_name + "-m" + std::to_string(m) + "-n" + std::to_string(n) + "-seed" + std::to_string(seed) + ".matrix";
	const std::string file_path = "./matrices/" + mat_file_name;
	std::ifstream ifs(file_path, std::ios::binary);
	if (!ifs) {
		return 0;
	}

	ifs.close();
	return 1;
}

void mtk::rsvd_test::get_input_matrix(
	float *const ptr,
	const std::string input_matrix_name,
	const std::size_t m, const std::size_t n,
	const std::uint64_t seed
	) {
	const std::string mat_file_name = input_matrix_name + "-m" + std::to_string(m) + "-n" + std::to_string(n) + "-seed" + std::to_string(seed) + ".matrix";
	const std::string file_path = "./matrices/" + mat_file_name;

	try {
		mtk::matfile::load_dense(
			ptr, m,
			file_path
			);
		return;
	} catch(const std::exception &e) {
		std::fprintf(stderr, "%s (cought @%s(%u))\n", e.what(), __FILE__, __LINE__);
	}

	// When matrix file does not exist, generate it
	std::string matrix_name_base = "";
	if (input_matrix_name.find_first_of("-", 0) != std::string::npos) {
		matrix_name_base = input_matrix_name.substr(0, input_matrix_name.find_first_of("-", 0));
	} else {
		matrix_name_base = input_matrix_name;
	}

	const auto start_clock = std::chrono::system_clock::now();
	bool generated = false;
	if (matrix_name_base == "latms") {
		const auto rank_str = input_matrix_name.substr(input_matrix_name.find_first_of("-", 0) + 1);
		const auto rank = std::stoul(rank_str);

		mtk::rsvd_test::gen_latms_matrix(
			ptr, m,
			m, n,
			rank,
			seed
			);
		generated = true;
	}
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
	std::fprintf(stderr, "Generated %s [%e]\n", file_path.c_str(), elapsed_time);

	if (generated) {
		mtk::matfile::save_dense(
			m, n,
			ptr, m,
			file_path
			);
	}
}

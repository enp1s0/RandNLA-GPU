#include <input_matrix.hpp>
#include <matfile/matfile.hpp>

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
		std::printf("%s (cought @%s(%u))\n", e.what(), __FILE__, __LINE__);
	}

	// When matrix file does not exist, generate it
}

#include <iostream>
#include <memory>
#include <mpi.h>
#include <input_matrix.hpp>

constexpr unsigned max_log_m = 10;
constexpr unsigned max_log_n = 10;
constexpr unsigned n_tests = 10;

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (unsigned log_m = 5; log_m <= max_log_m; log_m++) {
		for (unsigned log_n = 5; log_n <= max_log_n; log_n++) {
			const auto m = 1lu << log_m;
			const auto n = 1lu << log_n;
			auto uptr = std::unique_ptr<float[]>(new float[m * n]);
			const auto max_log_k = std::min(log_m, log_n);
			for (unsigned log_k = 4; log_k <= max_log_k - 1; log_k++) {
				const std::size_t k = 1lu << log_k;

				const std::string matrix_name = "latms-" + std::to_string(k);
				if (!mtk::rsvd_test::exist_input_matrix(matrix_name, m, n, rank)) {
					mtk::rsvd_test::get_input_matrix(
						uptr.get(), matrix_name,
						m, n,
						rank
						);
				}
			}
		}
	}

	MPI_Finalize();
}

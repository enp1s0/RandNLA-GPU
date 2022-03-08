#include <iostream>
#include <memory>
#include <mpi.h>
#include <input_matrix.hpp>

constexpr unsigned max_log_m = 16;
constexpr unsigned max_log_n = 16;
constexpr unsigned n_tests = 10;

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	for (unsigned seed = rank; seed < n_tests; seed += nprocs) {
		for (unsigned log_m = 16; log_m <= max_log_m; log_m++) {
			for (unsigned log_n = 16; log_n <= max_log_n; log_n++) {
				const auto m = 1lu << log_m;
				const auto n = 1lu << log_n;
				auto uptr = std::unique_ptr<float[]>(new float[m * n]);
				const auto max_log_k = std::min<int>(log_m, log_n) - 6;
				for (auto log_k = std::min<int>(log_m, log_n) - 10; log_k <= max_log_k; log_k++) {
					if (log_k < 0) continue;
					const std::size_t k = 1lu << log_k;

					const std::string matrix_name = "latms-" + std::to_string(k);
					if (!mtk::rsvd_test::exist_input_matrix(matrix_name, m, n, seed)) {
						mtk::rsvd_test::get_input_matrix(
							uptr.get(), matrix_name,
							m, n,
							seed
							);
					} else {
						std::printf("%s for m=%lu, n=%lu, seed=%lu exists\n",
								   matrix_name.c_str(), m, n, seed);
					}
				}
			}
		}
	}

	MPI_Finalize();
}

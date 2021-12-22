#include <iostream>
#include <chrono>
#include <vector>
#include <rsvd_test.hpp>
#include <input_matrix.hpp>
#include <cutf/memory.hpp>
#include <cutf/cusolver.hpp>
#include <cutf/stream.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>

constexpr unsigned max_log_m = 10;
constexpr unsigned max_log_n = 10;
constexpr unsigned n_svdj_iter = 10;

namespace {
void print_csv_header() {
	std::printf("implementation,matrix,m,n,k,p,n_svdj_iter,residual,u_orthogonality,v_orthogonality,throughput,n_tests\n");
}
void evaluate(
		const std::string implementation_name,
		const std::string input_matrix_name,
		mtk::rsvd_test::rsvd_base& rsvd,
		const unsigned n_tests,
		cudaStream_t const cuda_stream
		) {
	std::printf("%s,%s,%u,%u,%u,%u,%u,",
			implementation_name.c_str(),
			input_matrix_name.c_str(),
			rsvd.get_m(),
			rsvd.get_n(),
			rsvd.get_k(),
			rsvd.get_p(),
			rsvd.get_n_svdj_iter()
			);
	const auto A_size = rsvd.get_m() * rsvd.get_n();
	const auto S_size = std::min(rsvd.get_m(), rsvd.get_n());
	const auto U_size = rsvd.get_m() * (rsvd.get_k() + rsvd.get_p());
	const auto V_size = rsvd.get_n() * (rsvd.get_k() + rsvd.get_p());

	auto A_ptr = cutf::memory::malloc_async<float>(A_size, cuda_stream);
	auto U_ptr = cutf::memory::malloc_async<float>(U_size, cuda_stream);
	auto S_ptr = cutf::memory::malloc_async<float>(S_size, cuda_stream);
	auto V_ptr = cutf::memory::malloc_async<float>(V_size, cuda_stream);

	rsvd.set_input_ptr(A_ptr);
	rsvd.set_output_ptr(U_ptr, S_ptr, V_ptr);

	auto hA_ptr = cutf::memory::malloc_host<float>(A_size);
	rsvd.prepare();

	auto elapsed_time_sum = 0.;
	std::vector<double> residual_list(n_tests);
	std::vector<double> u_orthogonality_list(n_tests);
	std::vector<double> v_orthogonality_list(n_tests);
	for (unsigned i = 0; i < n_tests; i++) {
		// Initialize input matrix
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
		mtk::rsvd_test::get_input_matrix(
				hA_ptr, input_matrix_name,
				rsvd.get_m(), rsvd.get_n(),
				i
				);
		cutf::memory::copy_async(A_ptr, hA_ptr, A_size, cuda_stream);
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

		try {
			cudaStreamSynchronize(cuda_stream);
			const auto start_clock = std::chrono::system_clock::now();
			rsvd.run();
			cudaStreamSynchronize(cuda_stream);
			const auto end_clock = std::chrono::system_clock::now();
			const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
			elapsed_time_sum += elapsed_time;
			// Calculate the residual and orthogonality

			residual_list[i] = mtk::mateval::cuda::residual_UxSxVt(
					rsvd.get_m(), rsvd.get_n(), rsvd.get_k(),
					mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major,
					U_ptr, rsvd.get_m(),
					S_ptr,
					V_ptr, rsvd.get_n(),
					A_ptr, rsvd.get_m()
					);
			u_orthogonality_list[i] = mtk::mateval::cuda::orthogonality(
					rsvd.get_m(), rsvd.get_k(),
					mtk::mateval::col_major,
					U_ptr, rsvd.get_m()
					);
			v_orthogonality_list[i] = mtk::mateval::cuda::orthogonality(
					rsvd.get_n(), rsvd.get_k(),
					mtk::mateval::col_major,
					V_ptr, rsvd.get_n()
					);
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

		} catch (const std::exception& e) {
			std::printf("%s\n", e.what());
		}
	}
	std::printf("%e,%e,%e,",
			mtk::mateval::utils::calc_mean_and_var(residual_list).first,
			mtk::mateval::utils::calc_mean_and_var(u_orthogonality_list).first,
			mtk::mateval::utils::calc_mean_and_var(v_orthogonality_list).first
			);
	std::printf("%e,", elapsed_time_sum / n_tests);

	rsvd.clean();
	cutf::memory::free_async<float>(A_ptr, cuda_stream);
	cutf::memory::free_async<float>(U_ptr, cuda_stream);
	cutf::memory::free_async<float>(S_ptr, cuda_stream);
	cutf::memory::free_async<float>(V_ptr, cuda_stream);
	cutf::memory::free_host<float>(hA_ptr);
	std::printf("%u\n", n_tests);
}
} // noname namespace

int main() {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

	print_csv_header();
	for (unsigned log_m = 5; log_m <= max_log_m; log_m++) {
		for (unsigned log_n = 5; log_n <= max_log_n; log_n++) {
			const auto max_log_k = std::min(log_m, log_n);
			for (unsigned log_k = 4; log_k <= max_log_k - 1; log_k++) {
				const auto m = 1u << log_m;
				const auto n = 1u << log_n;
				const auto k = 1u << log_k;
				const auto p = k;
				if (k + p > std::min(m, n)) {
					break;
				}

				mtk::rsvd_test::rsvd_cusolver rsvd_cusolver(
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, k, p, n_svdj_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get()
						);

				evaluate("cusolver", "latms-" + std::to_string(k), rsvd_cusolver, 10, *cuda_stream.get());
			}
		}
	}
}

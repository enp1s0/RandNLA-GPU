#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <rsvd_test.hpp>
#include <input_matrix.hpp>
#include <cutf/memory.hpp>
#include <cutf/cusolver.hpp>
#include <cutf/stream.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>
#include <matfile/matfile.hpp>
#include <fphistogram/fphistogram.hpp>
#include <lapacke.h>

constexpr unsigned min_log_m = 11;
constexpr unsigned max_log_m = 13;
constexpr unsigned log_m_interval = 2;
constexpr unsigned min_log_n = 11;
constexpr unsigned max_log_n = 13;
constexpr unsigned log_n_interval = 2;
constexpr unsigned n_tests = 10;
constexpr unsigned n_iter = 0;
using svd_t = mtk::rsvd_test::svd_qr;

#ifdef TIME_BREAKDOWN
constexpr unsigned additional_num_tests_for_time_breakdown = 20;
#endif


namespace {

void svd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt, double* work, lapack_int lwork) {
	LAPACKE_dgesvd_work(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
}

void svd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, float* a, lapack_int lda, float* s, float* u, lapack_int ldu, float* vt, lapack_int ldvt, float* work, lapack_int lwork) {
	LAPACKE_sgesvd_work(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
}

template <class T>
void get_singular_value(
		T* const s_ptr,
		const std::size_t m, const std::size_t n,
		T* const a_ptr, const std::size_t lda
		) {
	int lwork = -1;
	T* work = nullptr;
	T tmp;
	svd(LAPACK_COL_MAJOR, 'N', 'N', m, n, a_ptr, lda, s_ptr, nullptr, 1, nullptr, 1, &tmp, lwork);

	lwork = static_cast<int>(tmp);
	work = new T [lwork];
	svd(LAPACK_COL_MAJOR, 'N', 'N', m, n, a_ptr, lda, s_ptr, nullptr, 1, nullptr, 1, work, lwork);

	delete [] work;
}

std::vector<std::string> str_split(const std::string str, const char d) {
	std::vector<std::string> strings;
	std::stringstream ss(str);
	std::string s;
	while (getline(ss, s, d)) {
		if (s.length() != 0) {
			strings.push_back(s);
		}
	}
	return strings;
}
void print_csv_header() {
	std::printf("implementation,matrix,m,n,k,p,n_iter,residual,u_orthogonality,v_orthogonality,time,n_tests\n");
	std::fflush(stdout);
}
void evaluate(
		const std::string input_matrix_name,
		mtk::rsvd_test::rsvd_base& rsvd,
		const unsigned n_tests,
		cudaStream_t const cuda_stream
		) {
	std::printf("%s,%s,%u,%u,%u,%u,%u,",
			rsvd.get_name().c_str(),
			input_matrix_name.c_str(),
			rsvd.get_m(),
			rsvd.get_n(),
			rsvd.get_k(),
			rsvd.get_p(),
			rsvd.get_n_iter()
			);
	const auto A_size = rsvd.get_m() * rsvd.get_n();
	const auto S_size = std::min(rsvd.get_m(), rsvd.get_n());
	const auto U_size = rsvd.get_m() * rsvd.get_k();
	const auto V_size = rsvd.get_n() * rsvd.get_k();

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
			rsvd.disable_breakdown_measurement();
			cudaStreamSynchronize(cuda_stream);
			const auto start_clock = std::chrono::system_clock::now();
			rsvd.run();
			cudaStreamSynchronize(cuda_stream);
			const auto end_clock = std::chrono::system_clock::now();
			const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
			elapsed_time_sum += elapsed_time;

			residual_list[i] = mtk::mateval::cuda::residual_UxSxVt(
					rsvd.get_m(), rsvd.get_n(), rsvd.get_k(),
					mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major,
					U_ptr, rsvd.get_m(),
					S_ptr,
					V_ptr, rsvd.get_n(),
					hA_ptr, rsvd.get_m()
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

			rsvd.enable_breakdown_measurement();
#ifdef TIME_BREAKDOWN
			for (unsigned i = 0; i < additional_num_tests_for_time_breakdown; i++) {
				rsvd.run();
			}
#endif

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
	std::fflush(stdout);
}

void breakdown_eval() {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), *cuda_stream.get()));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream.get());
	mtk::shgemm::enable_kernel_level_fixing(shgemm_handle, mtk::shgemm::detail::P1);

	print_csv_header();
	for (unsigned log_m = min_log_n; log_m <= max_log_m; log_m += log_m_interval) {
		//for (unsigned log_n = min_log_n; log_n <= max_log_n; log_n++) {
		{
			const auto log_n = log_m;
			const auto max_log_k = std::min(log_m, log_n);
			for (unsigned log_k = 8; log_k <= max_log_k - 4; log_k++) {
				const auto m = 1u << log_m;
				const auto n = 1u << log_n;
				const auto k = 1u << log_k;
				const auto decomp_k = k;
				const auto p = decomp_k / 10;
				if (decomp_k + p > std::min(m, n)) {
					break;
				}

				const std::string matrix_name = "cauchy";
				
				svd_t svd(*cusolver_handle.get());
				{
					mtk::rsvd_test::random_projection_fp32 rand_proj_fp32(*cublas_handle.get());
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_fp32
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
					std::printf("# START human time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_fp32.get_name().c_str());
					rsvd_selfmade.print_time_breakdown();
					std::printf("# END human\n");
					std::printf("# START csv time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_fp32.get_name().c_str());
					rsvd_selfmade.print_time_breakdown(true);
					std::printf("# END csv\n");
				}
				{
					mtk::rsvd_test::random_projection_tf32 rand_proj_tf32(*cublas_handle.get());
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_tf32
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
					std::printf("# START human time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_tf32.get_name().c_str());
					rsvd_selfmade.print_time_breakdown();
					std::printf("# END human\n");
					std::printf("# START csv time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_tf32.get_name().c_str());
					rsvd_selfmade.print_time_breakdown(true);
					std::printf("# END csv\n");
				}
				{
					mtk::rsvd_test::random_projection_shgemm rand_proj_shgemm(shgemm_handle, mtk::shgemm::tf32);
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_shgemm
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
					std::printf("# START human time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_shgemm.get_name().c_str());
					rsvd_selfmade.print_time_breakdown();
					std::printf("# END human\n");
					std::printf("# START csv time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_shgemm.get_name().c_str());
					rsvd_selfmade.print_time_breakdown(true);
					std::printf("# END csv\n");
				}
				{
					mtk::rsvd_test::random_projection_shgemm rand_proj_shgemm(shgemm_handle, mtk::shgemm::fp16);
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_shgemm
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
					std::printf("# START human time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_shgemm.get_name().c_str());
					rsvd_selfmade.print_time_breakdown();
					std::printf("# END human\n");
					std::printf("# START csv time-breakdown-%s-%u-%u-%u-%u-%s\n", matrix_name.c_str(), m, n, decomp_k, p, rand_proj_shgemm.get_name().c_str());
					rsvd_selfmade.print_time_breakdown(true);
					std::printf("# END csv\n");
				}
			}
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}

void accuracy_test() {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), *cuda_stream.get()));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream.get());

	std::vector<std::string> matrix_list = {"latms"/*, "latms_sigmoid"*/};

	print_csv_header();
	for (const auto& matrix : matrix_list) {
		for (unsigned log_m = min_log_n; log_m <= max_log_m; log_m++) {
			//for (unsigned log_n = min_log_n; log_n <= max_log_n; log_n++) {
			{
				const auto log_n = log_m;
				const auto max_log_k = std::min(log_m, log_n);
				const auto m = 1u << log_m;
				const auto n = 1u << log_n;

				const unsigned rank = std::min(m, n) / 32;
				unsigned k = rank;
				const auto decomp_k = k;
				const auto p = 32;
				if (decomp_k + p > std::min(m, n)) {
					break;
				}

				const std::string matrix_name = matrix + "-" + std::to_string(rank);

				svd_t svd(*cusolver_handle.get());

#if defined(RUN_REFERENCE_FUNCTIONS) && !defined(TIME_BREAKDOWN)
				mtk::rsvd_test::rsvd_cusolver rsvd_cusolver(
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get()
						);
				evaluate(matrix_name, rsvd_cusolver, n_tests, *cuda_stream.get());
#endif

				{
					mtk::rsvd_test::random_projection_fp32 rand_proj_fp32(*cublas_handle.get());
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_fp32
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
				}
				{
					mtk::rsvd_test::random_projection_tf32 rand_proj_tf32(*cublas_handle.get());
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_tf32
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
				}
				{
					mtk::rsvd_test::random_projection_shgemm rand_proj_shgemm(shgemm_handle, mtk::shgemm::tf32);
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_shgemm
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
				}
				{
					mtk::rsvd_test::random_projection_shgemm rand_proj_shgemm(shgemm_handle, mtk::shgemm::fp16);
					mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
							*cublas_handle.get(),
							*cusolver_handle.get(),
							*cusolver_params.get(),
							m, n, decomp_k, p, n_iter,
							nullptr, m,
							nullptr, m,
							nullptr,
							nullptr, n,
							*cuda_stream.get(),
							svd,
							rand_proj_shgemm
							);
					evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
				}

#if defined(RUN_REFERENCE_FUNCTIONS) && !defined(TIME_BREAKDOWN)
				mtk::rsvd_test::svdj_cusolver svdj_cusolver(
						*cusolver_handle.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get()
						);
				evaluate(matrix_name, svdj_cusolver, n_tests, *cuda_stream.get());
#endif
			}
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}

void sparse_accuracy_test() {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), *cuda_stream.get()));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream.get());

	std::vector<std::string> matrix_list = {"linear", "exp"};
	const std::size_t mat_N = 1u << 12;

	const std::size_t m = mat_N;
	const std::size_t n = mat_N;
	const std::size_t decomp_k = mat_N / 16;

	print_csv_header();
	for (const auto& matrix : matrix_list) {
		for (int log_s_p = -10; log_s_p <= -3; log_s_p++) {
			const auto p = 32;

			const std::string matrix_name = "designed-" + matrix + "-" + std::to_string(decomp_k) + "-" + std::to_string(-log_s_p);

			svd_t svd(*cusolver_handle.get());

			const std::vector<int> mantissa_length_list = {0, 4};
			for (int ml : mantissa_length_list) {
				std::printf("cut_mantissa_%u-", ml);
				mtk::rsvd_test::random_projection_fp32 rand_proj_fp32(*cublas_handle.get(), ml);
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_fp32
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				mtk::rsvd_test::random_projection_discrete rand_proj_discrete(*cublas_handle.get(), "sqrt3", {-std::sqrt(3.f), 0.f, std::sqrt(3.f)}, {1.f, 4.f, 1.f});
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_discrete
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				const float D = std::min(m, n);
				const unsigned s = 1750;
				const float s_fp = s;
				mtk::rsvd_test::random_projection_discrete rand_proj_discrete(*cublas_handle.get(), "s" + std::to_string(s), {-std::sqrt(s_fp), 0.f, std::sqrt(s_fp)}, {1.f, 2.f * s_fp - 2, 1.f});
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_discrete
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}

void designed_accuracy_test() {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), *cuda_stream.get()));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream.get());

	std::vector<std::string> matrix_list = {"linear", "exp"};
	const std::size_t mat_N = 1u << 12;

	const std::size_t m = mat_N;
	const std::size_t n = mat_N;
	const std::size_t decomp_k = mat_N / 16;

	print_csv_header();
	for (const auto& matrix : matrix_list) {
		for (int log_s_p = -10; log_s_p <= -3; log_s_p++) {
			const auto p = 32;

			const std::string matrix_name = "designed-" + matrix + "-" + std::to_string(decomp_k) + "-" + std::to_string(-log_s_p);

			svd_t svd(*cusolver_handle.get());

#ifdef CUT_MANTISSA
			const std::vector<int> mantissa_length_list = {0, 4};
			for (int ml : mantissa_length_list) {
				std::printf("%u,", ml);
				mtk::rsvd_test::random_projection_fp32 rand_proj_fp32(*cublas_handle.get(), ml);
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_fp32
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			continue;
#endif

#ifdef DISCRETE_RAND_TEST
			{
				mtk::rsvd_test::random_projection_discrete rand_proj_discrete(*cublas_handle.get(), "signed1", {-1.f, 1.f}, {1.f, 1.f});
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_discrete
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				mtk::rsvd_test::random_projection_discrete rand_proj_discrete(*cublas_handle.get(), "sqrt3", {-std::sqrt(3.f), 0.f, std::sqrt(3.f)}, {1.f, 4.f, 1.f});
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_discrete
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				const float D = std::min(m, n);
				const unsigned s = 1750;
				const float s_fp = s;
				mtk::rsvd_test::random_projection_discrete rand_proj_discrete(*cublas_handle.get(), "s" + std::to_string(s), {-std::sqrt(s_fp), 0.f, std::sqrt(s_fp)}, {1.f, 2.f * s_fp - 2, 1.f});
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_discrete
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			continue;
#endif
			{
				mtk::rsvd_test::random_projection_fp32 rand_proj_fp32(*cublas_handle.get());
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_fp32
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				mtk::rsvd_test::random_projection_tf32 rand_proj_tf32(*cublas_handle.get());
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_tf32
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				mtk::rsvd_test::random_projection_shgemm rand_proj_shgemm(shgemm_handle, mtk::shgemm::tf32);
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_shgemm
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}
			{
				mtk::rsvd_test::random_projection_shgemm rand_proj_shgemm(shgemm_handle, mtk::shgemm::fp16);
				mtk::rsvd_test::rsvd_selfmade rsvd_selfmade(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get(),
						svd,
						rand_proj_shgemm
						);
				evaluate(matrix_name, rsvd_selfmade, n_tests, *cuda_stream.get());
			}

#if defined(RUN_REFERENCE_FUNCTIONS) && !defined(TIME_BREAKDOWN)
			mtk::rsvd_test::svdj_cusolver svdj_cusolver(
					*cusolver_handle.get(),
					m, n, decomp_k, p, n_iter,
					nullptr, m,
					nullptr, m,
					nullptr,
					nullptr, n,
					*cuda_stream.get()
					);
			evaluate(matrix_name, svdj_cusolver, n_tests, *cuda_stream.get());

			mtk::rsvd_test::svd_cusolver svd_cusolver(
					*cusolver_handle.get(),
					m, n, decomp_k, p, n_iter,
					nullptr, m,
					nullptr, m,
					nullptr,
					nullptr, n,
					*cuda_stream.get()
					);
			evaluate(matrix_name, svd_cusolver, n_tests, *cuda_stream.get());

			mtk::rsvd_test::rsvd_cusolver rsvd_cusolver(
					*cusolver_handle.get(),
					*cusolver_params.get(),
					m, n, decomp_k, p, n_iter,
					nullptr, m,
					nullptr, m,
					nullptr,
					nullptr, n,
					*cuda_stream.get()
					);
			evaluate(matrix_name, rsvd_cusolver, n_tests, *cuda_stream.get());
#endif
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}

void watermark_core(
		mtk::rsvd_test::rsvd_base& rsvd,
		const std::string output_dir,
		const std::string base_name,
		const float* const u_ptr,
		const float* const s_ptr,
		const float* const v_ptr
		) {
	rsvd.prepare();
	rsvd.run();
	cudaDeviceSynchronize();

	const auto m = rsvd.get_m();
	const auto n = rsvd.get_n();
	const auto decomp_k = rsvd.get_k();

	mtk::matfile::save_dense(decomp_k, 1, s_ptr, decomp_k, output_dir + "/" + base_name + "." + rsvd.get_name() + ".s.matrix");
	mtk::matfile::save_dense(m, decomp_k, u_ptr, m,        output_dir + "/" + base_name + "." + rsvd.get_name() + ".u.matrix");
	mtk::matfile::save_dense(n, decomp_k, v_ptr, n,        output_dir + "/" + base_name + "." + rsvd.get_name() + ".v.matrix");

	std::printf("[%s] Largest sv = %e\n", rsvd.get_name().c_str(), s_ptr[0]);

	rsvd.clean();
}

void watermark(
		const std::string list_file_name,
		const std::string output_dir,
		const std::size_t max_image_width,
		const std::size_t max_image_height
		) {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), *cuda_stream.get()));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream.get());


	auto image_matrix_uptr = cutf::memory::get_host_unique_ptr<float>(max_image_height * max_image_width * 3);

	const auto max_rank = std::min(max_image_height, max_image_width * 3);

	auto s_uptr = cutf::memory::get_host_unique_ptr<float>(max_rank);
	auto u_uptr = cutf::memory::get_host_unique_ptr<float>(max_image_height * max_rank);
	auto v_uptr = cutf::memory::get_host_unique_ptr<float>(max_image_width * 3 * max_rank);

	svd_t svd(*cusolver_handle.get());

	std::ifstream ifs(list_file_name);
	std::string file_name;
	while (std::getline(ifs, file_name)) {
		std::size_t w, h;
		mtk::matfile::load_size(h, w, file_name);
		std::printf("file_name = %s\n", file_name.c_str());
		std::printf("image_matrix = (%lu x %lu)\n", w, h);
		std::fflush(stdout);

		const auto tmp_str_list = str_split(file_name, '/');
		for (const auto& s : tmp_str_list) std::printf("%s ", s.c_str());
		std::printf("\n");
		const auto tmp_str_list_2 = str_split(tmp_str_list[tmp_str_list.size() - 1], '.');
		auto base_name = tmp_str_list_2[0];
		for (unsigned i = 1; i < tmp_str_list_2.size() - 2; i++) {
			base_name += "." + tmp_str_list_2[i];
		}

		std::printf("base_name = %s\n", base_name.c_str());
		std::fflush(stdout);


		const auto m = h;
		const auto n = w;
		const auto p = 100lu;
		const auto decomp_k = std::stoul(tmp_str_list_2[tmp_str_list_2.size() - 2]);

		std::printf("input=(%lu, %lu), k = %lu, p = %lu\n", m, n, decomp_k, p);
		std::fflush(stdout);

		mtk::matfile::load_dense(image_matrix_uptr.get(), h, file_name);
		mtk::fphistogram::print_histogram<float, mtk::fphistogram::mode_log10>(image_matrix_uptr.get(), max_image_height * max_image_width);
		printf("(2,1) = [[%e], [%e]]\n", image_matrix_uptr.get()[0], image_matrix_uptr.get()[1]);

		// RSVD
		{
			mtk::rsvd_test::random_projection_fp32 rand_proj(*cublas_handle.get());
			mtk::rsvd_test::rsvd_selfmade rsvd(
					*cublas_handle.get(),
					*cusolver_handle.get(),
					*cusolver_params.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get(),
					svd,
					rand_proj
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
		{
			mtk::rsvd_test::random_projection_tf32 rand_proj(*cublas_handle.get());
			mtk::rsvd_test::rsvd_selfmade rsvd(
					*cublas_handle.get(),
					*cusolver_handle.get(),
					*cusolver_params.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get(),
					svd,
					rand_proj
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
		{
			mtk::rsvd_test::random_projection_shgemm rand_proj(shgemm_handle, mtk::shgemm::fp16);
			mtk::rsvd_test::rsvd_selfmade rsvd(
					*cublas_handle.get(),
					*cusolver_handle.get(),
					*cusolver_params.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get(),
					svd,
					rand_proj
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
		{
			mtk::rsvd_test::random_projection_shgemm rand_proj(shgemm_handle, mtk::shgemm::tf32);
			mtk::rsvd_test::rsvd_selfmade rsvd(
					*cublas_handle.get(),
					*cusolver_handle.get(),
					*cusolver_params.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get(),
					svd,
					rand_proj
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
		{
			mtk::rsvd_test::rsvd_cusolver rsvd(
					*cusolver_handle.get(),
					*cusolver_params.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get()
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
		{
			mtk::rsvd_test::svdj_cusolver rsvd(
					*cusolver_handle.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get()
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
		{
			mtk::rsvd_test::svd_cusolver rsvd(
					*cusolver_handle.get(),
					m, n, decomp_k, p, n_iter,
					image_matrix_uptr.get(), m,
					u_uptr.get(), m,
					s_uptr.get(),
					v_uptr.get(), n,
					*cuda_stream.get()
					);

			// load
			watermark_core(rsvd, output_dir, base_name, u_uptr.get(), s_uptr.get(), v_uptr.get());
		}
	}
}

void image_decomp_core(
		mtk::rsvd_test::rsvd_base& rsvd,
		float* const input_ptr,
		const float* const input_host_ptr,
		float* const u_ptr,
		float* const s_ptr,
		float* const v_ptr
		) {
	rsvd.set_input_ptr(input_ptr);
	rsvd.set_output_ptr(u_ptr, s_ptr, v_ptr);
	cudaDeviceSynchronize();
	rsvd.prepare();
	cudaDeviceSynchronize();
	rsvd.run();
	cudaDeviceSynchronize();

	const auto m = rsvd.get_m();
	const auto n = rsvd.get_n();
	const auto decomp_k = rsvd.get_k();

	//std::printf("[%s] Largest sv = %e\n", rsvd.get_name().c_str(), s_ptr[0]);
	const auto residual = mtk::mateval::cuda::residual_UxSxVt(
			rsvd.get_m(), rsvd.get_n(), rsvd.get_k(),
			mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major,
			u_ptr, rsvd.get_m(),
			s_ptr,
			v_ptr, rsvd.get_n(),
			input_host_ptr, rsvd.get_m()
			);
	const auto u_orthogonality = mtk::mateval::cuda::orthogonality(
			rsvd.get_m(), rsvd.get_k(),
			mtk::mateval::col_major,
			u_ptr, rsvd.get_m()
			);
	const auto v_orthogonality = mtk::mateval::cuda::orthogonality(
			rsvd.get_n(), rsvd.get_k(),
			mtk::mateval::col_major,
			v_ptr, rsvd.get_n()
			);
	std::printf("%s,%lu,%lu,%lu,%e,%e,%e\n",
			rsvd.get_name().c_str(),
			m, n, decomp_k,
		   	residual, u_orthogonality, v_orthogonality);
	std::fflush(stdout);

	cudaDeviceSynchronize();
	rsvd.clean();
}

void image_decomp(
		const std::string list_file_name,
		const std::size_t max_image_width,
		const std::size_t max_image_height
		) {
	auto cuda_stream  = cutf::stream::get_stream_unique_ptr();
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle.get(), *cuda_stream.get()));
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), *cuda_stream.get()));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream.get());

	auto host_image_matrix_uptr = cutf::memory::get_host_unique_ptr<float>(max_image_height * max_image_width);
	auto image_matrix_uptr = cutf::memory::get_device_unique_ptr<float>(max_image_height * max_image_width);

	const auto max_rank = std::max(max_image_width, max_image_height);

	auto s_list_matrix = cutf::memory::get_host_unique_ptr<float>(max_rank);

	auto host_s_uptr = cutf::memory::get_host_unique_ptr<float>(max_rank);
	auto host_u_uptr = cutf::memory::get_host_unique_ptr<float>(max_image_height * max_rank);
	auto host_v_uptr = cutf::memory::get_host_unique_ptr<float>(max_image_width * max_rank);

	auto s_uptr = cutf::memory::get_device_unique_ptr<float>(max_rank);
	auto u_uptr = cutf::memory::get_device_unique_ptr<float>(max_image_height * max_rank);
	auto v_uptr = cutf::memory::get_device_unique_ptr<float>(max_image_width * max_rank);

	svd_t svd(*cusolver_handle.get());

	std::ifstream ifs(list_file_name);
	std::string file_name;
	while (std::getline(ifs, file_name)) {
		std::size_t w, h;
		mtk::matfile::load_size(h, w, file_name);
		std::printf("file_name = %s\n", file_name.c_str());
		std::printf("image_matrix = (%lu x %lu)\n", w, h);
		std::fflush(stdout);

		const auto m = h;
		const auto n = w;
		const auto p = 10lu;

		mtk::matfile::load_dense(host_image_matrix_uptr.get(), h, file_name);

		double norm = 0.0f;
#pragma omp parallel for reduction(+:norm)
		for (std::size_t i = 0; i < h * w; i++) {
			norm += host_image_matrix_uptr.get()[i] * host_image_matrix_uptr.get()[i];
		}
		norm = std::sqrt(norm);

		cutf::memory::copy(image_matrix_uptr.get(), host_image_matrix_uptr.get(), w * h);

		auto A_dp_uptr = std::unique_ptr<double[]>(new double[m * n]);
#pragma omp parallel for
		for (std::size_t i = 0; i < m * n; i++) {
			A_dp_uptr.get()[i] = host_image_matrix_uptr.get()[i];
		}
		std::printf("Compute singular values ...\n");
		std::fflush(stdout);
		std::vector<double> s_computed(std::min(m, n));
		get_singular_value(s_computed.data(), m, n, A_dp_uptr.get(), m);

		for (int i = 0; i < s_computed.size(); i++) {
			s_computed[i] /= norm;
		}

		double sum_sp = 0;
		for (int i = s_computed.size() - 1; i >= 0; i--) {
			const double sum_sp_prev = sum_sp;
			sum_sp += s_computed.data()[i] * s_computed.data()[i];
			if (sum_sp_prev != 0 && static_cast<int>(std::log10(std::sqrt(sum_sp_prev))) != static_cast<int>(std::log10(std::sqrt(sum_sp)))) {
				std::printf("rank = %d, error = %e\n", i, std::sqrt(sum_sp));
			}
		}

		cudaDeviceSynchronize();

		for (unsigned mlog_s = 3; mlog_s <= 10; mlog_s++) {
			cutf::memory::copy(image_matrix_uptr.get(), host_image_matrix_uptr.get(), w * h);

			const auto designed_error = std::pow<float>(10.f, -static_cast<float>(mlog_s));
			std::size_t decomp_k = n-1;
			double p_rank_error = 0.0;
			for (; std::sqrt(p_rank_error) < designed_error; p_rank_error += [](const double x) -> double {return x * x;}(s_computed.data()[decomp_k--])){};

			std::printf("input=(%lu, %lu), k = %lu, p = %lu, theoretical_error = %e, (%e)\n", m, n, decomp_k, p, std::sqrt(p_rank_error) / norm, designed_error);
			std::printf("mode,m,n,rank,residual,v_orth,u_orth\n");
			std::fflush(stdout);

			// RSVD
			{
				mtk::rsvd_test::random_projection_fp32 rand_proj(*cublas_handle.get());
				mtk::rsvd_test::rsvd_selfmade rsvd(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						image_matrix_uptr.get(), m,
						u_uptr.get(), m,
						s_uptr.get(),
						v_uptr.get(), n,
						*cuda_stream.get(),
						svd,
						rand_proj
						);

				// load
				image_decomp_core(rsvd, image_matrix_uptr.get(), host_image_matrix_uptr.get(), u_uptr.get(), s_uptr.get(), v_uptr.get());
			}
			{
				mtk::rsvd_test::random_projection_tf32 rand_proj(*cublas_handle.get());
				mtk::rsvd_test::rsvd_selfmade rsvd(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						image_matrix_uptr.get(), m,
						u_uptr.get(), m,
						s_uptr.get(),
						v_uptr.get(), n,
						*cuda_stream.get(),
						svd,
						rand_proj
						);

				// load
				image_decomp_core(rsvd, image_matrix_uptr.get(), host_image_matrix_uptr.get(), u_uptr.get(), s_uptr.get(), v_uptr.get());
			}
			try {
				mtk::rsvd_test::random_projection_shgemm rand_proj(shgemm_handle, mtk::shgemm::fp16);
				mtk::rsvd_test::rsvd_selfmade rsvd(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						image_matrix_uptr.get(), m,
						u_uptr.get(), m,
						s_uptr.get(),
						v_uptr.get(), n,
						*cuda_stream.get(),
						svd,
						rand_proj
						);

				// load
				image_decomp_core(rsvd, image_matrix_uptr.get(), host_image_matrix_uptr.get(), u_uptr.get(), s_uptr.get(), v_uptr.get());
			} catch(const std::exception& e){std::cout<<e.what()<<std::endl;}
			try {
				mtk::rsvd_test::random_projection_shgemm rand_proj(shgemm_handle, mtk::shgemm::tf32);
				mtk::rsvd_test::rsvd_selfmade rsvd(
						*cublas_handle.get(),
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, decomp_k, p, n_iter,
						image_matrix_uptr.get(), m,
						u_uptr.get(), m,
						s_uptr.get(),
						v_uptr.get(), n,
						*cuda_stream.get(),
						svd,
						rand_proj
						);

				// load
				image_decomp_core(rsvd, image_matrix_uptr.get(), host_image_matrix_uptr.get(), u_uptr.get(), s_uptr.get(), v_uptr.get());
			} catch(const std::exception& e){std::cout<<e.what()<<std::endl;}
			{
				mtk::rsvd_test::svd_cusolver rsvd(
						*cusolver_handle.get(),
						m, n, decomp_k, p, n_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n,
						*cuda_stream.get()
						);

				// load
				image_decomp_core(rsvd, image_matrix_uptr.get(), host_image_matrix_uptr.get(), u_uptr.get(), s_uptr.get(), v_uptr.get());
			}
		}
	}
}
} // namespace

int main(int argc, char** argv) {
	if (argc == 4 && std::string(argv[1]) == "watermark") {
		watermark(argv[2], argv[3], 4032, 4032);
	} else if (argc == 3 && std::string(argv[1]) == "image") {
		image_decomp(argv[2], 5000, 5000);
	} else if (argc == 2 && std::string(argv[1]) == "breakdown") {
		breakdown_eval();
	} else if (argc == 2 && std::string(argv[1]) == "designed") {
		designed_accuracy_test();
	} else if (argc == 2 && std::string(argv[1]) == "sparse") {
		sparse_accuracy_test();
	} else {
		accuracy_test();
	}
}

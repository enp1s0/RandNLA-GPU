#include <iostream>
#include <chrono>
#include <cutf/cutensor.hpp>
#include <cutf/stream.hpp>
#include <cutf/memory.hpp>
#include <mateval/comparison_cuda.hpp>
#include "hosvd_test.hpp"
#include "eval.hpp"

constexpr unsigned num_mode = 3;
constexpr unsigned min_tensor_dim_log = 9;
constexpr unsigned max_tensor_dim_log = 10;
constexpr unsigned min_rank_log = 6;
constexpr unsigned num_throughput_test = 1u << 4;

void test_hosvd(
		const cuta::mode_t& input_tensor_mode,
		const cuta::mode_t& core_tensor_mode,
		mtk::rsvd_test::hosvd_base& hosvd,
		cudaStream_t cuda_stream
		) {
	// memory allocations
	const auto input_size = cuta::utils::get_num_elements(input_tensor_mode);
	const auto S_size = cuta::utils::get_num_elements(core_tensor_mode);

	auto A_ptr = cutf::memory::malloc_async<float>(input_size, cuda_stream);
	auto S_ptr = cutf::memory::malloc_async<float>(S_size, cuda_stream);

	std::vector<float*> Q_ptrs(input_tensor_mode.size());
	std::vector<cuta::mode_t> Q_modes(input_tensor_mode.size());
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		const auto malloc_size = input_tensor_mode[i].second * core_tensor_mode[i].second;
		Q_ptrs[i] = cutf::memory::malloc_async<float>(malloc_size, cuda_stream);
		Q_modes[i].push_back(input_tensor_mode[i]);
		Q_modes[i].push_back(core_tensor_mode[i]);
	}

	hosvd.set_config(A_ptr, S_ptr, Q_ptrs);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	hosvd.prepare();

	// tensor elements initialization
	mtk::rsvd_test::gen_input_tensor(
			hosvd.get_cutensor_handle(),
			A_ptr,
			S_ptr, core_tensor_mode,
			Q_ptrs, Q_modes,
			hosvd.get_work_mem_ptr(),
			"random",
			cuda_stream
			);

	auto host_A_uptr = cutf::memory::get_host_unique_ptr<float>(input_size);
	cutf::memory::copy_async(host_A_uptr.get(), A_ptr, input_size, cuda_stream);

	// accuracy test
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	hosvd.run();
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

	mtk::rsvd_test::contract(
			hosvd.get_cutensor_handle(),
			A_ptr,
			S_ptr,
			core_tensor_mode,
			Q_ptrs,
			Q_modes,
			hosvd.get_work_mem_ptr(),
			cuda_stream
			);

	// calculate error
	cutf::memory::copy_async(hosvd.get_work_mem_ptr(), host_A_uptr.get(), input_size, cuda_stream);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

	const auto error = mtk::mateval::cuda::get_error(
			mtk::mateval::relative_residual,
			input_size, 1,
			mtk::mateval::col_major,
			mtk::mateval::col_major,
			hosvd.get_work_mem_ptr(), input_size,
			A_ptr, input_size
			);
	std::vector<double> orthogonalities(input_tensor_mode.size());
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		orthogonalities[i] = mtk::mateval::cuda::orthogonality(
				Q_modes[i][0].second, Q_modes[i][1].second,
				mtk::mateval::col_major,
				Q_ptrs[i], Q_modes[i][0].second
				);
	}

	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

	// throughput test
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	const auto start_clock = std::chrono::system_clock::now();
	for (unsigned i = 0; i < num_throughput_test; i++)
		hosvd.run();
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / num_throughput_test;

	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	cutf::memory::free_async(A_ptr, cuda_stream);
	cutf::memory::free_async(S_ptr, cuda_stream);
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		cutf::memory::free_async(Q_ptrs[i], cuda_stream);
	}
	hosvd.clean();

	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	// Output result
	std::printf("%s,", hosvd.get_name_str().c_str());
	std::printf("(");
	for (const auto i : input_tensor_mode) {
		std::printf("%lu-", i.second);
	}
	std::printf("),");
	std::fflush(stdout);
	std::printf("(");
	for (const auto i : core_tensor_mode) {
		std::printf("%lu-", i.second);
	}
	std::printf("),");

	std::printf("%e,", elapsed_time);
	std::printf("%e,", error.at(mtk::mateval::relative_residual));
	for (const auto o : orthogonalities) {
		std::printf("%e,", o);
	}
	std::printf("\n");
}

int main() {
	auto cusolver_handle_uptr = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cublas_handle_uptr   = cutf::cublas::get_cublas_unique_ptr();
	auto cuda_stream_uptr = cutf::stream::get_stream_unique_ptr();
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle_uptr.get(), *cuda_stream_uptr.get()));
	CUTF_CHECK_ERROR(cusolverDnSetStream(*cusolver_handle_uptr.get(), *cuda_stream_uptr.get()));

	cutensorHandle_t cutensor_handle;
	CUTF_HANDLE_ERROR(cutensorInit(&cutensor_handle));

	mtk::shgemm::shgemmHandle_t shgemm_handle;
	mtk::shgemm::create(shgemm_handle);
	mtk::shgemm::set_cuda_stream(shgemm_handle, *cuda_stream_uptr.get());

	for (unsigned tensor_dim_log = min_tensor_dim_log; tensor_dim_log <= max_tensor_dim_log; tensor_dim_log++) {
		const unsigned max_rank_log = tensor_dim_log - 1;
		for (unsigned rank_log = min_rank_log; rank_log <= max_rank_log; rank_log++) {
			const auto rank = 1u << rank_log;
			const auto dim = 1u << tensor_dim_log;

			cuta::mode_t input_tensor_mode;
			cuta::mode_t core_tensor_mode;
			for (unsigned i = 0; i < num_mode; i++) {
				cuta::utils::insert_mode(input_tensor_mode, "m-" + std::to_string(i), dim);
				cuta::utils::insert_mode(core_tensor_mode , "c-" + std::to_string(i), rank);
			}

			{
				mtk::rsvd_test::random_projection_fp32 rp(*cublas_handle_uptr.get());
				mtk::rsvd_test::hosvd_rp hosvd(
						input_tensor_mode,
						core_tensor_mode,
						rp,
						*cuda_stream_uptr.get(),
						*cusolver_handle_uptr.get(),
						cutensor_handle
						);
				test_hosvd(
						input_tensor_mode,
						core_tensor_mode,
						hosvd,
						*cuda_stream_uptr.get()
						);
#ifdef TIME_BREAKDOWN
				std::printf("# START human time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown();
				std::printf("# END human\n");
				std::printf("# START csv time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown(true);
				std::printf("# END csv\n");
#endif
			}

			{
				mtk::rsvd_test::random_projection_tf32 rp(*cublas_handle_uptr.get());
				mtk::rsvd_test::hosvd_rp hosvd(
						input_tensor_mode,
						core_tensor_mode,
						rp,
						*cuda_stream_uptr.get(),
						*cusolver_handle_uptr.get(),
						cutensor_handle
						);
				test_hosvd(
						input_tensor_mode,
						core_tensor_mode,
						hosvd,
						*cuda_stream_uptr.get()
						);
#ifdef TIME_BREAKDOWN
				std::printf("# START human time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown();
				std::printf("# END human\n");
				std::printf("# START csv time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown(true);
				std::printf("# END csv\n");
#endif
			}

			{
				mtk::rsvd_test::random_projection_shgemm rp(shgemm_handle, mtk::shgemm::tf32);
				mtk::rsvd_test::hosvd_rp hosvd(
						input_tensor_mode,
						core_tensor_mode,
						rp,
						*cuda_stream_uptr.get(),
						*cusolver_handle_uptr.get(),
						cutensor_handle
						);
				test_hosvd(
						input_tensor_mode,
						core_tensor_mode,
						hosvd,
						*cuda_stream_uptr.get()
						);
#ifdef TIME_BREAKDOWN
				std::printf("# START human time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown();
				std::printf("# END human\n");
				std::printf("# START csv time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown(true);
				std::printf("# END csv\n");
#endif
			}
			{
				mtk::rsvd_test::random_projection_shgemm rp(shgemm_handle, mtk::shgemm::fp16);
				mtk::rsvd_test::hosvd_rp hosvd(
						input_tensor_mode,
						core_tensor_mode,
						rp,
						*cuda_stream_uptr.get(),
						*cusolver_handle_uptr.get(),
						cutensor_handle
						);
				test_hosvd(
						input_tensor_mode,
						core_tensor_mode,
						hosvd,
						*cuda_stream_uptr.get()
						);
#ifdef TIME_BREAKDOWN
				std::printf("# START human time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown();
				std::printf("# END human\n");
				std::printf("# START csv time-breakdown-%s-%u-%u\n", hosvd.get_name_str().c_str(), dim, rank);
				hosvd.print_time_breakdown(true);
				std::printf("# END csv\n");
#endif
			}
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}

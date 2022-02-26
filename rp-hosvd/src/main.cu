#include <iostream>
#include <chrono>
#include <cutf/cutensor.hpp>
#include <cutf/stream.hpp>
#include <cutf/memory.hpp>
#include "hosvd_test.hpp"

constexpr unsigned num_mode = 3;
constexpr unsigned tensor_dim_log = 9;
constexpr unsigned min_rank_log = 3;
constexpr unsigned max_rank_log = tensor_dim_log - 1;
constexpr unsigned num_throughput_test = 1u << 4;

void test_hosvd(
		const cutt::mode_t& input_tensor_mode,
		const cutt::mode_t& core_tensor_mode,
		mtk::rsvd_test::hosvd_base& hosvd,
		cudaStream_t cuda_stream
		) {
	// memory allocations
	const auto input_size = cutt::utils::get_num_elements(input_tensor_mode);
	const auto S_size = cutt::utils::get_num_elements(core_tensor_mode);

	auto A_ptr = cutf::memory::malloc_async<float>(input_size, cuda_stream);
	auto S_ptr = cutf::memory::malloc_async<float>(S_size, cuda_stream);

	std::vector<float*> Q_ptrs(input_tensor_mode.size());
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		const auto malloc_size = input_tensor_mode[i].second * core_tensor_mode[i].second;
		Q_ptrs[i] = cutf::memory::malloc_async<float>(malloc_size, cuda_stream);
	}

	hosvd.set_config(A_ptr, S_ptr, Q_ptrs);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	hosvd.prepare();

	// tensor elements initialization

	// accuracy test
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	hosvd.run();
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

	for (unsigned rank_log = min_rank_log; rank_log <= max_rank_log; rank_log++) {
		const auto rank = 1u << rank_log;
		const auto dim = 1u << tensor_dim_log;

		cutt::mode_t input_tensor_mode;
		cutt::mode_t core_tensor_mode;
		for (unsigned i = 0; i < num_mode; i++) {
			cutt::utils::insert_mode(input_tensor_mode, "m-" + std::to_string(i), dim);
			cutt::utils::insert_mode(core_tensor_mode , "c-" + std::to_string(i), rank);
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
		}
	}
	mtk::shgemm::destroy(shgemm_handle);
}

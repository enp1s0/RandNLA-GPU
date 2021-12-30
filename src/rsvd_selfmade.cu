#include <rsvd_test.hpp>
#include <cutf/type.hpp>
#include <cutf/curand.hpp>

void mtk::rsvd_test::rsvd_selfmade::prepare() {
	const auto q = get_k() + get_p();

	int tmp_work_size;

	// MATMUL(RAND)
	working_memory.rand_matrix_size = get_n() * q;
	working_memory.y_matrix_size = get_m() * q;

	// QR
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
				cusolver_handle,
				get_m(), q,
				working_memory.alloc_ptr, get_m(),
				&tmp_work_size
				));
	working_memory.geqrf_0_size = tmp_work_size;
	//working_memory.tau_size = std::min(get_m(), q);
	// To share the memory with full_S, the sizee of tau is q.
	working_memory.tau_size = q;

	CUTF_CHECK_ERROR(cusolverDnSorgqr_bufferSize(
				cusolver_handle,
				get_m(), q,
				q,
				working_memory.y_matrix_ptr, get_m(),
				working_memory.tau_ptr,
				&tmp_work_size
				));
	working_memory.orgqr_0_size = tmp_work_size;

	// MATMUL (B)
	working_memory.b_matrix_size = get_n() * q;

	// SVDJ
	constexpr double tol = 1e-7;
	CUTF_CHECK_ERROR(cusolverDnCreateGesvdjInfo(&svdj_params));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetMaxSweeps(svdj_params, get_n_svdj_iter()));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetTolerance(svdj_params, tol));
	CUTF_CHECK_ERROR(cusolverDnSgesvdj_bufferSize(
				cusolver_handle,
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				q, get_n(),
				working_memory.b_matrix_ptr, get_n(),
				S_ptr,
				working_memory.small_u_ptr, q,
				working_memory.full_V_ptr, get_n(),
				&tmp_work_size,
				svdj_params
				));
	working_memory.gesvdj_size = tmp_work_size;
	working_memory.small_u_size = q * q;
	working_memory.full_V_size = q * get_n();
	working_memory.full_S_size = q;

	// Memory allocation
	const std::size_t cusolver_working_memory_size = std::max(std::max(working_memory.geqrf_0_size, working_memory.orgqr_0_size), working_memory.gesvdj_size);
	const std::size_t tmp_matrix_size = working_memory.rand_matrix_size + working_memory.y_matrix_size + working_memory.tau_size + working_memory.b_matrix_size + working_memory.full_V_size + working_memory.small_u_size;

	// Allocate
	working_memory.alloc_ptr = cutf::memory::malloc_async<float>(cusolver_working_memory_size + tmp_matrix_size, cuda_stream);

	// Split
	working_memory.rand_mat_ptr = working_memory.alloc_ptr;
	working_memory.y_matrix_ptr = working_memory.rand_mat_ptr + working_memory.rand_matrix_size;
	working_memory.b_matrix_ptr = working_memory.y_matrix_ptr + working_memory.y_matrix_size;
	working_memory.tau_ptr = working_memory.b_matrix_ptr + working_memory.b_matrix_size;
	working_memory.full_V_ptr = working_memory.tau_ptr + working_memory.tau_size;
	working_memory.full_S_ptr = working_memory.tau_ptr;
	working_memory.small_u_ptr = working_memory.full_V_ptr + working_memory.full_V_size;
	working_memory.geqrf_0_ptr = working_memory.small_u_ptr + working_memory.small_u_size;
	working_memory.gesvdj_ptr = working_memory.geqrf_0_ptr;
	working_memory.orgqr_0_ptr = working_memory.geqrf_0_ptr;

	// DevInfo
	working_memory.devInfo_ptr = cutf::memory::malloc_async<int>(1, cuda_stream);
}

void mtk::rsvd_test::rsvd_selfmade::run() {
	const auto q = get_k() + get_p();

	// generate random matrix
	const uint64_t seed = 10;
	const float alpha = 1.f, beta = 0.f;
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(curandSetStream(*cugen.get(), cuda_stream));

#ifdef TIME_BREAKDOWN
	profiler.start_timer_sync("gen_rand");
#endif
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), working_memory.rand_mat_ptr, working_memory.rand_matrix_size));

#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("gen_rand");
	profiler.start_timer_sync("matmul_1");
#endif
	CUTF_CHECK_ERROR(cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				get_m(), q, get_n(),
				&alpha,
				A_ptr, get_m(),
				working_memory.rand_mat_ptr, get_n(),
				&beta,
				working_memory.y_matrix_ptr, get_m()
				));
#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("matmul_1");
	profiler.start_timer_sync("qr");
#endif
	// QR(1)
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
				cusolver_handle,
				get_m(), q,
				working_memory.y_matrix_ptr, get_m(),
				working_memory.tau_ptr,
				working_memory.geqrf_0_ptr,
				working_memory.geqrf_0_size,
				working_memory.devInfo_ptr
				));
	CUTF_CHECK_ERROR(cusolverDnSorgqr(
				cusolver_handle,
				get_m(), q,
				q,
				working_memory.y_matrix_ptr, get_m(),
				working_memory.tau_ptr,
				working_memory.orgqr_0_ptr,
				working_memory.orgqr_0_size,
				working_memory.devInfo_ptr
				));

	// MATMUL(2)
#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("qr");
	profiler.start_timer_sync("matmul_2");
#endif
	CUTF_CHECK_ERROR(cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				q, get_n(), get_m(),
				&alpha,
				working_memory.y_matrix_ptr, get_m(),
				A_ptr, get_m(),
				&beta,
				working_memory.b_matrix_ptr, q
				));

	// SVD
#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("matmul_2");
	profiler.start_timer_sync("svd");
#endif
	CUTF_CHECK_ERROR(cusolverDnSgesvdj(
				cusolver_handle,
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				q, get_n(),
				working_memory.b_matrix_ptr, q,
				working_memory.full_S_ptr,
				working_memory.small_u_ptr, q,
				working_memory.full_V_ptr, get_n(),
				working_memory.gesvdj_ptr,
				working_memory.gesvdj_size,
				working_memory.devInfo_ptr,
				svdj_params
				));
#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("svd");
	profiler.start_timer_sync("matmul_3");
#endif
	CUTF_CHECK_ERROR(cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				get_m(), k, q,
				&alpha,
				working_memory.y_matrix_ptr, get_m(),
				working_memory.small_u_ptr, q,
				&beta,
				U_ptr, ldu
				));
#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("matmul_3");
	profiler.start_timer_sync("matmul_copy");
#endif
	mtk::rsvd_test::copy_matrix(
			get_n(), get_k(),
			V_ptr, ldv,
			working_memory.full_V_ptr, get_n(),
			cuda_stream
			);
	mtk::rsvd_test::copy_matrix(
			get_k(), 1,
			S_ptr, get_k(),
			working_memory.full_S_ptr, q,
			cuda_stream
			);
#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("matmul_copy");
#endif
}

void mtk::rsvd_test::rsvd_selfmade::clean() {
	cutf::memory::free_async(working_memory.devInfo_ptr, cuda_stream);
	cutf::memory::free_async(working_memory.alloc_ptr, cuda_stream);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	CUTF_CHECK_ERROR(cusolverDnDestroyGesvdjInfo(svdj_params));
}

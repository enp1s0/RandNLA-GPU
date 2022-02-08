#include <rsvd_test.hpp>
#include <cutf/type.hpp>
#include <cutf/curand.hpp>

namespace {
#ifdef FP16_EMULATION
__global__ void fp16_emulation_kernel(
		float* const ptr,
		const std::size_t size
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= size) {
		return;
	}

	ptr[tid] = cutf::type::cast<float>(cutf::type::cast<half>(ptr[tid]));
}

void fp16_emulation(
		float* const ptr,
		const std::size_t size,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	const auto grid_size = (size + block_size - 1) / block_size;

	fp16_emulation_kernel<<<grid_size, block_size, 0, cuda_stream>>>(ptr, size);
}
#endif // FP16_EMULATION

__global__ void power_iteration_singular_value_root_kernel (
		float* const dst_s_array_ptr,
		const float* const src_s_array_ptr,
		const std::size_t s_array_size,
		const int num_iter
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= s_array_size) {
		return;
	}
	dst_s_array_ptr[tid] = powf(src_s_array_ptr[tid], 1.f / (2 * num_iter + 1));
}

void power_iteration_singular_value_root(
		float* const dst_s_array_ptr,
		const float* const src_s_array_ptr,
		const std::size_t s_array_size,
		const int num_iter,
		cudaStream_t cuda_stream
		) {
	constexpr unsigned block_size = 256;
	const unsigned grid_size = (s_array_size + block_size - 1) / block_size;

	power_iteration_singular_value_root_kernel<<<block_size, grid_size, 1, cuda_stream>>>(
			dst_s_array_ptr,
			src_s_array_ptr,
			s_array_size,
			num_iter
			);
}
} // noname namespace

void mtk::rsvd_test::rsvd_selfmade::prepare() {
	const auto q = get_k() + get_p();

	if (get_n_iter() == 0) {
		throw std::runtime_error("n_iter must be >= 1");
	}

	int tmp_work_size;

	// MATMUL(RAND)
	rand_proj.set_config(
			get_m(), get_n(), q,
			cuda_stream
			);
	rand_proj.allocate_working_memory();
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
	if (svd.get_name_str() == "svd_jaccobi") {
		svd.prepare(q, get_n());
	} else {
		svd.prepare(get_n(), q);
	}
	working_memory.gesvdj_size = svd.get_working_mem_size();
	working_memory.small_u_size = q * q;
	working_memory.full_V_size = q * get_n();
	working_memory.full_S_size = q;

	// For power iteration
	working_memory.bbt_size = q * q;

	// Memory allocation
	const std::size_t cusolver_working_memory_size = std::max(std::max(working_memory.geqrf_0_size, working_memory.orgqr_0_size), working_memory.gesvdj_size);
	const std::size_t tmp_matrix_size = working_memory.y_matrix_size + working_memory.tau_size + working_memory.b_matrix_size + working_memory.full_V_size + working_memory.small_u_size;
	const std::size_t iter_working_size = (get_n_iter() == 0) ? 0lu : working_memory.b_matrix_size + working_memory.bbt_size;

	// Allocate
	working_memory.alloc_ptr = cutf::memory::malloc_async<float>(cusolver_working_memory_size + tmp_matrix_size + iter_working_size, cuda_stream);

	// Split
	working_memory.y_matrix_ptr = working_memory.alloc_ptr;
	working_memory.b_matrix_ptr = working_memory.y_matrix_ptr + working_memory.y_matrix_size;
	working_memory.tau_ptr = working_memory.b_matrix_ptr + working_memory.b_matrix_size;
	working_memory.full_V_ptr = working_memory.tau_ptr + working_memory.tau_size;
	working_memory.full_S_ptr = working_memory.tau_ptr;
	working_memory.small_u_ptr = working_memory.full_V_ptr + working_memory.full_V_size;
	if (get_n_iter() == 0) {
		working_memory.geqrf_0_ptr = working_memory.small_u_ptr + working_memory.small_u_size;
		working_memory.b_2_ptr = working_memory.b_matrix_ptr;
	} else {
		working_memory.bbt_1_ptr = working_memory.small_u_ptr;
		working_memory.bbt_2_ptr = working_memory.small_u_ptr + working_memory.bbt_size;
		working_memory.b_2_ptr = working_memory.bbt_2_ptr + working_memory.bbt_size;
		working_memory.geqrf_0_ptr = working_memory.b_2_ptr + working_memory.b_matrix_size;
	}
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

#ifdef TIME_BREAKDOWN
	profiler.start_timer_sync("gen_rand");
#endif

	rand_proj.gen_rand(seed);

#ifdef TIME_BREAKDOWN
	profiler.stop_timer_sync("gen_rand");
	profiler.start_timer_sync("matmul_1");
#endif
	rand_proj.apply(
				working_memory.y_matrix_ptr, get_m(),
				A_ptr, get_m()
			);
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
#endif

	if (get_n_iter()) {
#ifdef TIME_BREAKDOWN
		profiler.start_timer_sync("power_iter");
#endif
		CUTF_CHECK_ERROR(cutf::cublas::gemm(
					cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_T,
					q, q, get_n(),
					&alpha,
					working_memory.b_matrix_ptr, q,
					working_memory.b_matrix_ptr, q,
					&beta,
					working_memory.bbt_1_ptr, q
					));
		for (unsigned i = 1; i < get_n_iter(); i++) {
			const float* const input_ptr  = ((i & 0x1) == 0) ? working_memory.bbt_2_ptr : working_memory.bbt_1_ptr;
			float* const       output_ptr = ((i & 0x1) == 1) ? working_memory.bbt_2_ptr : working_memory.bbt_1_ptr;
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						q, q, q,
						&alpha,
						input_ptr, q,
						input_ptr, q,
						&beta,
						output_ptr, q
						));
		}
		const float* const bbt_ptr = ((get_n_iter() & 0x1) == 0) ? working_memory.bbt_2_ptr : working_memory.bbt_1_ptr;
		if (svd.get_name_str() == "svd_jaccobi") {
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						q, get_n(), q,
						&alpha,
						bbt_ptr, q,
						working_memory.b_matrix_ptr, q,
						&beta,
						working_memory.b_2_ptr, q
						));
		} else {
			// When using svd_qr, the input matrix must be transposed because it must be tall.
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						cublas_handle,
						CUBLAS_OP_T, CUBLAS_OP_T,
						get_n(), q, q,
						&alpha,
						working_memory.b_matrix_ptr, q,
						bbt_ptr, q,
						&beta,
						working_memory.b_2_ptr, get_n()
						));
		}
#ifdef TIME_BREAKDOWN
		profiler.stop_timer_sync("power_iter");
#endif
	}
	if (svd.get_name_str() == "svd_jaccobi") {
#ifdef TIME_BREAKDOWN
		profiler.start_timer_sync("svd");
#endif
		const auto svd_ldv = get_n();
		svd.run(
				working_memory.full_S_ptr,
				working_memory.small_u_ptr, q,
				working_memory.full_V_ptr, svd_ldv,
				working_memory.b_2_ptr, q,
				working_memory.gesvdj_ptr
				);
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
				working_memory.full_V_ptr, svd_ldv,
				cuda_stream
				);
#ifdef TIME_BREAKDOWN
		profiler.stop_timer_sync("matmul_copy");
#endif
	} else {
#ifdef TIME_BREAKDOWN
		profiler.start_timer_sync("svd");
#endif
		const auto svd_ldv = get_n();
		svd.run(
				working_memory.full_S_ptr,
				working_memory.full_V_ptr, svd_ldv,
				working_memory.small_u_ptr, q,
				working_memory.b_2_ptr, get_n(),
				working_memory.gesvdj_ptr
				);
#ifdef TIME_BREAKDOWN
		profiler.stop_timer_sync("svd");
		profiler.start_timer_sync("matmul_3");
#endif
		CUTF_CHECK_ERROR(cutf::cublas::gemm(
					cublas_handle,
					CUBLAS_OP_N, CUBLAS_OP_T,
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
				working_memory.full_V_ptr, svd_ldv,
				cuda_stream
				);
#ifdef TIME_BREAKDOWN
		profiler.stop_timer_sync("matmul_copy");
#endif
	}
#ifdef TIME_BREAKDOWN
	profiler.start_timer_sync("adjust_s");
#endif
	// Fix singular values
	if (get_n_iter()) {
		power_iteration_singular_value_root(
				S_ptr,
				working_memory.full_S_ptr,
				get_k(),
				get_n_iter(),
				cuda_stream
				);
	} else {
		mtk::rsvd_test::copy_matrix(
				get_k(), 1,
				S_ptr, get_k(),
				working_memory.full_S_ptr, q,
				cuda_stream
				);
	}
#ifdef TIME_BREAKDOWN
		profiler.stop_timer_sync("adjust_s");
#endif
}

void mtk::rsvd_test::rsvd_selfmade::clean() {
	cutf::memory::free_async(working_memory.devInfo_ptr, cuda_stream);
	cutf::memory::free_async(working_memory.alloc_ptr, cuda_stream);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
	svd.free();
	rand_proj.free_working_memory();
}

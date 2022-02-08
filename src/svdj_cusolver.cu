#include <rsvd_test.hpp>
#include <cutf/type.hpp>

void mtk::rsvd_test::svdj_cusolver::prepare() {
	const auto r = std::min(get_m(), get_n());
	full_S_ptr = cutf::memory::malloc_async<float>(r, cuda_stream);
	full_U_ptr = cutf::memory::malloc_async<float>(r * get_m(), cuda_stream);
	full_V_ptr = cutf::memory::malloc_async<float>(r * get_n(), cuda_stream);

	// SVDJ
	constexpr double tol = 1e-7;
	constexpr unsigned num_svdj_iter = 20;
	CUTF_CHECK_ERROR(cusolverDnCreateGesvdjInfo(&svdj_params));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetMaxSweeps(svdj_params, num_svdj_iter));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetTolerance(svdj_params, tol));

	int tmp_working_memory_size;
	CUTF_CHECK_ERROR(cusolverDnSgesvdj_bufferSize(
				cusolver_handle,
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				get_m(), get_n(),
				A_ptr, lda,
				full_S_ptr,
				full_U_ptr, get_m(),
				full_V_ptr, get_n(),
				&tmp_working_memory_size,
				svdj_params
				));

	working_memory_device_size = tmp_working_memory_size;
	working_memory_device_ptr = cutf::memory::malloc_async<float>(working_memory_device_size, cuda_stream);

	devInfo_ptr = cutf::memory::malloc_async<int>(1, cuda_stream);
}

void mtk::rsvd_test::svdj_cusolver::run() {
	CUTF_CHECK_ERROR(cusolverDnSgesvdj(
				cusolver_handle,
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				get_m(), get_n(),
				A_ptr, get_m(),
				full_S_ptr,
				full_U_ptr, get_m(),
				full_V_ptr, get_n(),
				working_memory_device_ptr,
				working_memory_device_size,
				devInfo_ptr,
				svdj_params
				));
	mtk::rsvd_test::copy_matrix(
			get_m(), get_k(),
			U_ptr, ldu,
			full_U_ptr, get_m(),
			cuda_stream
			);
	mtk::rsvd_test::copy_matrix(
			get_n(), get_k(),
			V_ptr, ldv,
			full_V_ptr, get_n(),
			cuda_stream
			);
	mtk::rsvd_test::copy_matrix(
			get_k(), 1,
			S_ptr, get_k(),
			full_S_ptr, get_n(),
			cuda_stream
			);
}

void mtk::rsvd_test::svdj_cusolver::clean() {
	cutf::memory::free_async(devInfo_ptr, cuda_stream);
	cutf::memory::free_async(full_S_ptr, cuda_stream);
	cutf::memory::free_async(full_U_ptr, cuda_stream);
	cutf::memory::free_async(full_V_ptr, cuda_stream);
	cutf::memory::free_async(working_memory_device_ptr, cuda_stream);
}

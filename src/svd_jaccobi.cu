#include "svd_base.hpp"
#include <cutf/cusolver.hpp>

std::size_t mtk::rsvd_test::svd_jaccobi::get_working_mem_size() {
	int tmp_work_size;
	CUTF_CHECK_ERROR(cusolverDnSgesvdj_bufferSize(
				cusolver_handle,
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				m, n,
				nullptr, n,
				nullptr,
				nullptr, m,
				nullptr, n,
				&tmp_work_size,
				svdj_params
				));

	work_size = tmp_work_size + 1;

	return work_size;
}

void mtk::rsvd_test::svd_jaccobi::prepare(const std::size_t im, const std::size_t in) {
	m = im;
	n = in;
	constexpr double tol = 1e-6;
	constexpr unsigned num_svdj_iter = 20;
	CUTF_CHECK_ERROR(cusolverDnCreateGesvdjInfo(&svdj_params));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetMaxSweeps(svdj_params, num_svdj_iter));
	CUTF_CHECK_ERROR(cusolverDnXgesvdjSetTolerance(svdj_params, tol));
}

void mtk::rsvd_test::svd_jaccobi::run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr) {

	CUTF_CHECK_ERROR(cusolverDnSgesvdj(
				cusolver_handle,
				CUSOLVER_EIG_MODE_VECTOR,
				1,
				m, n,
				input_ptr, ld,
				S_ptr,
				U_ptr, ldu,
				V_ptr, ldv,
				work_ptr,
				work_size,
				reinterpret_cast<int*>(work_ptr + work_size),
				svdj_params
				));

}

void mtk::rsvd_test::svd_jaccobi::free() {
	CUTF_CHECK_ERROR(cusolverDnDestroyGesvdjInfo(svdj_params));
}

#include "svd_base.hpp"
#include <cutf/cusolver.hpp>

std::size_t mtk::rsvd_test::svd_qr::get_working_mem_size() {
	int l_work;
	CUTF_CHECK_ERROR(cusolverDnSgesvd_bufferSize(cusolver_handle, m, n, &l_work));

	const std::size_t r_work = std::min(m, n) - 1;

	work_size = (l_work + r_work) + 1;

	return work_size;
}

void mtk::rsvd_test::svd_qr::run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr) {

	const std::size_t r_work = std::min(m, n) - 1;
	const std::size_t l_work = work_size - r_work - 1;
	int* devInfo = reinterpret_cast<int*>(work_ptr + l_work + r_work);
	CUTF_CHECK_ERROR(cusolverDnSgesvd(
				cusolver_handle,
				'S', 'S',
				m, n,
				input_ptr, ld,
				S_ptr,
				U_ptr, ldu,
				V_ptr, ldv,
				work_ptr,
				l_work,
				work_ptr + l_work,
				devInfo
				));
}

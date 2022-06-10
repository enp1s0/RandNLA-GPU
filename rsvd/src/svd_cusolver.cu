#include <iostream>
#include <rsvd_test.hpp>
#include <stdexcept>
#include <cuda_common.hpp>
#include <cutf/type.hpp>

void mtk::rsvd_test::svd_cusolver::prepare() {
	const auto r = std::min(get_m(), get_n());
	full_S_ptr = cutf::memory::malloc_async<float>(r, cuda_stream);
	full_U_ptr = cutf::memory::malloc_async<float>(r * get_m(), cuda_stream);
	full_Vt_ptr = cutf::memory::malloc_async<float>(r * get_n(), cuda_stream);

	// SVD
	int l_work;
	CUTF_CHECK_ERROR(cusolverDnSgesvd_bufferSize(
				cusolver_handle,
				get_m(), get_n(),
				&l_work
				));

	const std::size_t r_work = std::min(get_m(), get_n()) - 1;

	working_memory_device_size = l_work + r_work;
	working_memory_device_ptr = cutf::memory::malloc_async<float>(working_memory_device_size, cuda_stream);

	devInfo_ptr = cutf::memory::malloc_async<int>(1, cuda_stream);
}

void mtk::rsvd_test::svd_cusolver::run() {
	const std::size_t r_work = std::min(get_m(), get_n()) - 1;
	const std::size_t l_work = working_memory_device_size - r_work;
	if (get_n() > get_m()) {
		throw std::runtime_error("gesvd only supports m>=n");
	}
	CUTF_CHECK_ERROR(cusolverDnSgesvd(
				cusolver_handle,
				'S', 'S',
				get_m(), get_n(),
				A_ptr, get_m(),
				full_S_ptr,
				full_U_ptr, get_m(),
				full_Vt_ptr, get_n(),
				working_memory_device_ptr,
				working_memory_device_size,
				working_memory_device_ptr + l_work,
				devInfo_ptr
				));
	mtk::rsvd_test::copy_matrix(
			get_m(), get_k(),
			U_ptr, ldu,
			full_U_ptr, get_m(),
			cuda_stream
			);
	mtk::rsvd_test::transpose_matrix(
			get_n(), get_k(),
			V_ptr, ldv,
			full_Vt_ptr, get_n(),
			cuda_stream
			);
	mtk::rsvd_test::copy_matrix(
			get_k(), 1,
			S_ptr, get_k(),
			full_S_ptr, get_n(),
			cuda_stream
			);
}

void mtk::rsvd_test::svd_cusolver::clean() {
	cutf::memory::free_async(devInfo_ptr, cuda_stream);
	cutf::memory::free_async(full_S_ptr, cuda_stream);
	cutf::memory::free_async(full_U_ptr, cuda_stream);
	cutf::memory::free_async(full_Vt_ptr, cuda_stream);
	cutf::memory::free_async(working_memory_device_ptr, cuda_stream);
}

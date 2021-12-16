#include <rsvd_test.hpp>
#include <cutf/type.hpp>

void mtk::rsvd_test::rsvd_cusolver::prepare() {
	CUTF_CHECK_ERROR(cusolverDnXgesvdr_bufferSize(
				cusolver_handle,
				cusolver_params,
				'S', 'S',
				m, n, k, p,
				n_svdj_iter,
				cutf::type::get_data_type<float>(),
				A_ptr, lda,
				cutf::type::get_data_type<float>(),
				S_ptr,
				cutf::type::get_data_type<float>(),
				U_ptr, ldu,
				cutf::type::get_data_type<float>(),
				Vt_ptr, ldvt,
				cutf::type::get_data_type<float>(),
				&working_memory_device_size,
				&working_memory_host_size
				));
	working_memory_host_uptr = cutf::memory::get_host_unique_ptr<uint8_t>(working_memory_host_size);
	working_memory_device_uptr = cutf::memory::get_device_unique_ptr<uint8_t>(working_memory_device_size);
	devInfo_uptr = cutf::memory::get_device_unique_ptr<int>(1);
}

void mtk::rsvd_test::rsvd_cusolver::run() {
	CUTF_CHECK_ERROR(cusolverDnXgesvdr(
					cusolver_handle,
					cusolver_params,
					'S', 'S',
					m, n, k, p,
					n_svdj_iter,
					cutf::type::get_data_type<float>(),
					A_ptr, lda,
					cutf::type::get_data_type<float>(),
					S_ptr,
					cutf::type::get_data_type<float>(),
					U_ptr, ldu,
					cutf::type::get_data_type<float>(),
					Vt_ptr, ldvt,
					cutf::type::get_data_type<float>(),
					working_memory_device_uptr.get(),
					working_memory_device_size,
					working_memory_host_uptr.get(),
					working_memory_host_size,
					devInfo_uptr.get()
				));
}

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
				V_ptr, ldv,
				cutf::type::get_data_type<float>(),
				&working_memory_device_size,
				&working_memory_host_size
				));
	working_memory_host_ptr = cutf::memory::malloc_host<uint8_t>(working_memory_host_size);
	working_memory_device_ptr = cutf::memory::malloc_async<uint8_t>(working_memory_device_size, cuda_stream);
	devInfo_ptr = cutf::memory::malloc_async<int>(1, cuda_stream);
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
					V_ptr, ldv,
					cutf::type::get_data_type<float>(),
					working_memory_device_ptr,
					working_memory_device_size,
					working_memory_host_ptr,
					working_memory_host_size,
					devInfo_ptr
				));
}

void mtk::rsvd_test::rsvd_cusolver::clean() {
	cutf::memory::free_async(working_memory_device_ptr, cuda_stream);
	cutf::memory::free_async(devInfo_ptr, cuda_stream);
	cutf::memory::free_host(working_memory_host_ptr);
}

#include <cutf/memory.hpp>
#include <cutt/reshape.hpp>
#include <cuda_common.hpp>
#include "hosvd_test.hpp"

void mtk::rsvd_test::hosvd_rp::prepare() {
	std::size_t prod = 1;
	for (const unsigned i : mode) prod *= i;
	std::size_t max_rand_matrix_m = 0;
	std::size_t max_rand_matrix_n = 0;
	std::size_t max_rand_matrix_size = 0;
	std::size_t max_rand_matrix_index = 0;
	for (unsigned i = 0; i < mode.size(); i++) {
		const auto rand_m = prod / mode[i];
		const auto rand_n = rank[i];

		if (rand_m * rand_n > max_rand_matrix_size) {
			max_rand_matrix_index = i;
			max_rand_matrix_m = rand_m;
			max_rand_matrix_n = rand_n;
			max_rand_matrix_size = rand_m * rand_n;
		}
	}

	random_projection.set_config(
			mode[max_rand_matrix_index], max_rand_matrix_m, max_rand_matrix_n
			);
	random_projection.allocate_working_memory();

	// For TTGT
	working_memory.ttgt_size = prod;

	// QR
	for (unsigned i = 0; i < mode.size(); i++) {
		int qr_size_0, qr_size_1;
		CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
					cusolver_handle,
					mode[i], rank[i],
					Q_ptr[i], mode[i],
					&qr_size_0
					));

		CUTF_CHECK_ERROR(cusolverDnSorgqr_bufferSize(
					cusolver_handle,
					mode[i], rank[i],
					rank[i],
					Q_ptr[i], mode[i],
					working_memory.tau_ptr,
					&qr_size_1
					));
		working_memory.geqrf_size.push_back(qr_size_0);
		working_memory.orgqr_size.push_back(qr_size_1);
		working_memory.tau_size = std::max<std::size_t>(working_memory.tau_size, rank[i]);
		working_memory.qr_size  = std::max<std::size_t>(working_memory.qr_size , std::max(qr_size_0, qr_size_1)) + 1;
	}

	working_memory.alloc_size = working_memory.ttgt_size + working_memory.qr_size + working_memory.tau_size + 1 /*dev*/;

	working_memory.alloc_ptr = cutf::memory::malloc_async<float>(working_memory.alloc_size, cuda_stream);

	working_memory.ttgt_ptr = working_memory.alloc_ptr;
	working_memory.qr_ptr   = working_memory.ttgt_ptr + working_memory.ttgt_size;
	working_memory.tau_ptr  = working_memory.qr_ptr + working_memory.qr_size;
	working_memory.dev_ptr  = reinterpret_cast<int*>(working_memory.tau_ptr) + working_memory.tau_size;
}

void mtk::rsvd_test::hosvd_rp::clean() {
	cutf::memory::free_async(working_memory.alloc_ptr, cuda_stream);
	random_projection.free_working_memory();
}

void mtk::rsvd_test::hosvd_rp::run() {
	std::size_t dim_product = 1;
	cutt::mode_t original_mode;
	for (unsigned i = 0; i < mode.size(); i++) {
		cutt::utils::insert_mode(original_mode, "m-" + std::to_string(i), mode[i]);
		dim_product *= mode[i];
	}
	// Transpose the tensor
	for (unsigned i = 0; i < mode.size(); i++) {
		std::vector<std::string> reshaped_mode_order(original_mode.size());
		const auto target_mode_name = "m-" + std::to_string(i);
		reshaped_mode_order[0] = target_mode_name;
		for (unsigned j = 0, k = 1; j < mode.size(); j++) {
			if (i != j) {
				reshaped_mode_order[k++] = "m-" + std::to_string(j);
			}
		}
		// Transpose
		cutt::reshape(
				working_memory.ttgt_ptr,
				A_ptr,
				original_mode,
				reshaped_mode_order,
				cuda_stream
				);
		// Rand projection
		random_projection.apply(
				mode[i], rank[i], dim_product / mode[i],
				Q_ptr[i], mode[i],
				working_memory.ttgt_ptr, mode[i]
				);
		// QR
		CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
					cusolver_handle,
					mode[i], rank[i],
					working_memory.ttgt_ptr, mode[i],
					working_memory.tau_ptr,
					working_memory.qr_ptr,
					working_memory.geqrf_size[i],
					working_memory.dev_ptr
					));
		CUTF_CHECK_ERROR(cusolverDnSorgqr(
					cusolver_handle,
					mode[i], rank[i],
					rank[i],
					working_memory.ttgt_ptr, mode[i],
					working_memory.tau_ptr,
					working_memory.qr_ptr,
					working_memory.orgqr_size[i],
					working_memory.dev_ptr
					));
	}
}

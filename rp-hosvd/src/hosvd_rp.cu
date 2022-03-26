#include <cutf/memory.hpp>
#include <cutf/cutensor.hpp>
#include <cuta/cutensor_utils.hpp>
#include <cuda_common.hpp>
#include "hosvd_test.hpp"

#ifdef TIME_BREAKDOWN
#define CUTF_PROFILE_START_TIMER(name) CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));profiler.start_timer_sync(name)
#define CUTF_PROFILE_STOP_TIMER(name)  CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));profiler.stop_timer_sync(name)
#else
#define CUTF_PROFILE_START_TIMER(name)
#define CUTF_PROFILE_STOP_TIMER(name)
#endif

void mtk::rsvd_test::hosvd_rp::prepare() {
	working_memory.alloc_ptr = nullptr;
	contraction_working_mem_ptr = nullptr;

	const std::size_t prod = cuta::utils::get_num_elements(input_tensor_mode);
	std::size_t max_rand_matrix_m = 0;
	std::size_t max_rand_matrix_n = 0;
	std::size_t max_rand_matrix_size = 0;
	std::size_t max_rand_matrix_index = 0;
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		const auto rand_m = prod / input_tensor_mode[i].second;
		const auto rand_n = core_tensor_mode[i].second;

		if (rand_m * rand_n > max_rand_matrix_size) {
			max_rand_matrix_index = i;
			max_rand_matrix_m = rand_m;
			max_rand_matrix_n = rand_n;
			max_rand_matrix_size = rand_m * rand_n;
		}
	}

	random_projection.set_config(
			input_tensor_mode[max_rand_matrix_index].second, max_rand_matrix_m, max_rand_matrix_n,
			cuda_stream
			);
	random_projection.allocate_working_memory();

	// For TTGT
	working_memory.ttgt_size = prod;

	// QR
	Q_tensor_mode.resize                 (input_tensor_mode.size());
	Q_tensor_desc.resize                 (input_tensor_mode.size());
	Q_tensor_alignment_requirement.resize(input_tensor_mode.size());
	working_memory.tau_size = 0;
	working_memory.qr_size  = 0;
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		Q_tensor_mode[i].push_back(input_tensor_mode[i]);
		Q_tensor_mode[i].push_back(core_tensor_mode[i]);
		Q_tensor_desc[i] = cuta::cutensor::get_descriptor<float>(cutensor_handle, Q_tensor_mode[i]);
		CUTT_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, Q_ptr[i], &Q_tensor_desc[i], &Q_tensor_alignment_requirement[i]));

		int qr_size_0, qr_size_1;
		CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
					cusolver_handle,
					input_tensor_mode[i].second, core_tensor_mode[i].second,
					Q_ptr[i], input_tensor_mode[i].second,
					&qr_size_0
					));

		CUTF_CHECK_ERROR(cusolverDnSorgqr_bufferSize(
					cusolver_handle,
					input_tensor_mode[i].second, core_tensor_mode[i].second,
					core_tensor_mode[i].second,
					Q_ptr[i], input_tensor_mode[i].second,
					working_memory.tau_ptr,
					&qr_size_1
					));
		working_memory.geqrf_size.push_back(qr_size_0);
		working_memory.orgqr_size.push_back(qr_size_1);
		working_memory.tau_size = std::max<std::size_t>(working_memory.tau_size, core_tensor_mode[i].second);
		working_memory.qr_size  = std::max<std::size_t>(working_memory.qr_size , std::max(qr_size_0, qr_size_1)) + 1;
	}

	// Allocation (1)
	working_memory.alloc_size = working_memory.ttgt_size + working_memory.qr_size + working_memory.tau_size + 1 /*dev*/;

	working_memory.alloc_ptr = cutf::memory::malloc_async<float>(working_memory.alloc_size, cuda_stream);

	working_memory.ttgt_ptr = working_memory.alloc_ptr;
	working_memory.qr_ptr   = working_memory.ttgt_ptr + working_memory.ttgt_size;
	working_memory.tau_ptr  = working_memory.qr_ptr + working_memory.qr_size;
	working_memory.dev_ptr  = reinterpret_cast<int*>(working_memory.tau_ptr) + working_memory.tau_size;

	// Resize
	contraction_desc.resize            (input_tensor_mode.size());
	contraction_working_mem_size.resize(input_tensor_mode.size());
	contraction_find.resize            (input_tensor_mode.size());
	contraction_plan.resize            (input_tensor_mode.size());
	// Tensor contraction
	tmp_core_tensor_mode.resize                 (input_tensor_mode.size() + 1);
	tmp_core_tensor_desc.resize                 (input_tensor_mode.size() + 1);
	tmp_core_tensor_alignment_requirement.resize(input_tensor_mode.size() + 1);
	tmp_core_tensor_mode[0] = input_tensor_mode;
	tmp_core_tensor_desc[0] = cuta::cutensor::get_descriptor<float>(cutensor_handle, input_tensor_mode);
	CUTT_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, A_ptr, &tmp_core_tensor_desc[0], &tmp_core_tensor_alignment_requirement[0]));
	for (unsigned i = 1; i <= input_tensor_mode.size(); i++) {
		auto t_mode = tmp_core_tensor_mode[i - 1];
		t_mode[i - 1] = core_tensor_mode[i - 1];
		tmp_core_tensor_mode[i] = t_mode;

		float* mem_ptr;
		if (i == input_tensor_mode.size()) {
			mem_ptr = S_ptr;
		} else if (i % 2 == 0) {
			mem_ptr = A_ptr;
		} else {
			mem_ptr = working_memory.alloc_ptr;
		}
		tmp_core_tensor_desc[i] = cuta::cutensor::get_descriptor<float>(cutensor_handle, tmp_core_tensor_mode[i]);
		CUTT_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, mem_ptr, &tmp_core_tensor_desc[i], &tmp_core_tensor_alignment_requirement[i]));

		// Set contraction descriptor
		CUTT_CHECK_ERROR(cutensorInitContractionDescriptor(&cutensor_handle, &contraction_desc[i - 1],
				&tmp_core_tensor_desc[i - 1], cuta::cutensor::get_extent_list_in_int(tmp_core_tensor_mode[i - 1]).data(), tmp_core_tensor_alignment_requirement[i - 1],
				&Q_tensor_desc       [i - 1], cuta::cutensor::get_extent_list_in_int(Q_tensor_mode       [i - 1]).data(), Q_tensor_alignment_requirement       [i - 1],
				&tmp_core_tensor_desc[i    ], cuta::cutensor::get_extent_list_in_int(tmp_core_tensor_mode[i    ]).data(), tmp_core_tensor_alignment_requirement[i    ],
				&tmp_core_tensor_desc[i    ], cuta::cutensor::get_extent_list_in_int(tmp_core_tensor_mode[i    ]).data(), tmp_core_tensor_alignment_requirement[i    ],
				cuta::cutensor::get_compute_type<float>()));

		// Set find
		CUTT_CHECK_ERROR(cutensorInitContractionFind(&cutensor_handle, &contraction_find[i - 1], CUTENSOR_ALGO_DEFAULT));

		// calculate working memory size
		CUTT_CHECK_ERROR(cutensorContractionGetWorkspace(&cutensor_handle, &contraction_desc[i - 1], &contraction_find[i - 1], CUTENSOR_WORKSPACE_RECOMMENDED, &contraction_working_mem_size[i - 1]));

		// set plan
		CUTT_CHECK_ERROR(cutensorInitContractionPlan(&cutensor_handle, &contraction_plan[i - 1], &contraction_desc[i - 1], &contraction_find[i - 1], contraction_working_mem_size[i - 1]));
	}
	// Calc working memory size
	contraction_working_mem_size_max = 0;
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		contraction_working_mem_size_max = std::max(contraction_working_mem_size_max, contraction_working_mem_size[i]);
	}
	contraction_working_mem_ptr = cutf::memory::malloc_async<uint8_t>(contraction_working_mem_size_max, cuda_stream);

	random_projection.gen_rand(100);
}

void mtk::rsvd_test::hosvd_rp::clean() {
	cutf::memory::free_async(working_memory.alloc_ptr, cuda_stream);
	working_memory.alloc_ptr = nullptr;
	cutf::memory::free_async(contraction_working_mem_ptr, cuda_stream);
	contraction_working_mem_ptr = nullptr;
	random_projection.free_working_memory();
}

void mtk::rsvd_test::hosvd_rp::run() {
	const float alpha = 1.f;
	const float beta = 0.f;
	// Transpose the tensor
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		// Transpose
		std::vector<std::string> reshaped_mode_order(input_tensor_mode.size());
		const auto target_mode_name = input_tensor_mode[i].first;
		reshaped_mode_order[0] = target_mode_name;
		for (unsigned j = 0, k = 1; j < input_tensor_mode.size(); j++) {
			if (i != j) {
				reshaped_mode_order[k++] = input_tensor_mode[j].first;
			}
		}
		const auto permutated_mode = cuta::utils::get_permutated_mode(input_tensor_mode, reshaped_mode_order);
		const auto desc_A = cuta::cutensor::get_descriptor<float>(cutensor_handle, input_tensor_mode);
		const auto desc_B = cuta::cutensor::get_descriptor<float>(cutensor_handle, permutated_mode);
		CUTF_PROFILE_START_TIMER("reshape");
		//cuttExecute(cutt_handle_list[i], A_ptr, working_memory.ttgt_ptr);
		CUTF_CHECK_ERROR(cutensorPermutation(
					&cutensor_handle,
					&alpha,
					A_ptr,
					&desc_A,
					cuta::cutensor::get_extent_list_in_int(input_tensor_mode).data(),
					working_memory.ttgt_ptr,
					&desc_B,
					cuta::cutensor::get_extent_list_in_int(permutated_mode).data(),
					cuta::cutensor::get_data_type<float>(),
					cuda_stream
					));
		CUTF_PROFILE_STOP_TIMER("reshape");
		CUTF_PROFILE_START_TIMER("random_projection");
		// Rand projection
		random_projection.apply(
				input_tensor_mode[i].second, cuta::utils::get_num_elements(input_tensor_mode) / input_tensor_mode[i].second, core_tensor_mode[i].second,
				Q_ptr[i], input_tensor_mode[i].second,
				working_memory.ttgt_ptr, input_tensor_mode[i].second
				);
		CUTF_PROFILE_STOP_TIMER("random_projection");
		CUTF_PROFILE_START_TIMER("qr");
		// QR
		CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
					cusolver_handle,
					input_tensor_mode[i].second, core_tensor_mode[i].second,
					Q_ptr[i], input_tensor_mode[i].second,
					working_memory.tau_ptr,
					working_memory.qr_ptr,
					working_memory.geqrf_size[i],
					working_memory.dev_ptr
					));
		CUTF_CHECK_ERROR(cusolverDnSorgqr(
					cusolver_handle,
					input_tensor_mode[i].second, core_tensor_mode[i].second,
					core_tensor_mode[i].second,
					Q_ptr[i], input_tensor_mode[i].second,
					working_memory.tau_ptr,
					working_memory.qr_ptr,
					working_memory.orgqr_size[i],
					working_memory.dev_ptr
					));
		CUTF_PROFILE_STOP_TIMER("qr");
	}
	// Compute the core tensor
	float *input_ptr;
	float *output_ptr;
	for (unsigned i = 0; i < input_tensor_mode.size(); i++) {
		if (i % 2 == 0) {
			input_ptr = A_ptr;
		} else {
			input_ptr = working_memory.alloc_ptr;
		}
		if (i == input_tensor_mode.size() - 1) {
			output_ptr = S_ptr;
		} else if (i % 2 == 0) {
			output_ptr = working_memory.alloc_ptr;
		} else {
			output_ptr = A_ptr;
		}
		CUTF_PROFILE_START_TIMER("tensor_contraction");
		CUTT_CHECK_ERROR(cutensorContraction(&cutensor_handle,
				&contraction_plan[i],
				reinterpret_cast<const void*>(&alpha), input_ptr, Q_ptr[i],
				reinterpret_cast<const void*>(&beta), output_ptr, output_ptr,
				contraction_working_mem_ptr, contraction_working_mem_size[i], 0
				));
		CUTF_PROFILE_STOP_TIMER("tensor_contraction");
	}
}

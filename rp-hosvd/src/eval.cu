#include <cassert>
#include <cutf/cutensor.hpp>
#include <cutf/error.hpp>
#include <cutf/memory.hpp>
#include <cuta/cutensor_utils.hpp>
#include "eval.hpp"

void mtk::rsvd_test::contract (
		cutensorHandle_t cutensor_handle,
		float* const out_ptr,
		const float* const core_tensor_ptr,
		cuta::mode_t core_tensor_mode,
		const std::vector<float*> q_matrices,
		const std::vector<cuta::mode_t> q_matrix_modes,
		float* const work_ptr,
		cudaStream_t cuda_stream
		) {
	assert(q_matrices.size() == q_matrix_modes.size());

	cuta::mode_t current_output_mode = core_tensor_mode;
	float *output_tensor_ptrs[2];
	if (core_tensor_mode.size() % 2 == 0) {
		output_tensor_ptrs[0] = work_ptr;
		output_tensor_ptrs[1] = out_ptr;
	} else {
		output_tensor_ptrs[0] = out_ptr;
		output_tensor_ptrs[1] = work_ptr;
	}

	const float* input_tensor_ptr = core_tensor_ptr;
	for (unsigned i = 0; i < q_matrices.size(); i++) {
		assert(q_matrix_modes[i].size() == 2);
		cuta::mode_t output_mode;
		for (const auto& mode : current_output_mode) {
			if (mode.first == q_matrix_modes[i][0].first) {
				output_mode.push_back(q_matrix_modes[i][1]);
			} else if (mode.first == q_matrix_modes[i][1].first) {
				output_mode.push_back(q_matrix_modes[i][0]);
			} else {
				output_mode.push_back(mode);
			}
		}
		const auto desc_A = cuta::cutensor::get_descriptor<float>(cutensor_handle, current_output_mode);
		const auto desc_B = cuta::cutensor::get_descriptor<float>(cutensor_handle, q_matrix_modes[i]);
		const auto desc_C = cuta::cutensor::get_descriptor<float>(cutensor_handle, output_mode);

		uint32_t alignment_requirement_A;
		CUTF_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, input_tensor_ptr, &desc_A, &alignment_requirement_A));

		uint32_t alignment_requirement_B;
		CUTF_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, q_matrices[i], &desc_B, &alignment_requirement_B));

		uint32_t alignment_requirement_C;
		CUTF_CHECK_ERROR(cutensorGetAlignmentRequirement(&cutensor_handle, output_tensor_ptrs[i % 2], &desc_C, &alignment_requirement_C));

		cutensorContractionDescriptor_t desc_contraction;
		CUTF_CHECK_ERROR(cutensorInitContractionDescriptor(&cutensor_handle, &desc_contraction,
					&desc_A, cuta::cutensor::get_extent_list_in_int(current_output_mode).data(), alignment_requirement_A,
					&desc_B, cuta::cutensor::get_extent_list_in_int(q_matrix_modes[i]).data(), alignment_requirement_B,
					&desc_C, cuta::cutensor::get_extent_list_in_int(output_mode).data(), alignment_requirement_C,
					&desc_C, cuta::cutensor::get_extent_list_in_int(output_mode).data(), alignment_requirement_C,
					cutf::cutensor::get_compute_type<float>()));

		cutensorContractionFind_t find;
		CUTF_CHECK_ERROR(cutensorInitContractionFind(&cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));

		std::size_t work_size = 0;
		CUTF_CHECK_ERROR(cutensorContractionGetWorkspace(&cutensor_handle, &desc_contraction, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &work_size));

		auto workspace = cutf::memory::malloc_async<uint8_t>(work_size, cuda_stream);
		void* const workspace_ptr = workspace;

		cutensorContractionPlan_t plan;
		CUTF_CHECK_ERROR(cutensorInitContractionPlan(&cutensor_handle, &plan, &desc_contraction, &find, work_size));

		const float alpha = 1.0f;
		const float beta = 0.0f;

		CUTF_CHECK_ERROR(cutensorContraction(&cutensor_handle,
					&plan,
					reinterpret_cast<const void*>(&alpha), input_tensor_ptr, q_matrices[i],
					reinterpret_cast<const void*>(&beta), output_tensor_ptrs[i % 2], output_tensor_ptrs[i % 2],
					workspace_ptr, work_size, 0
					));

		current_output_mode = output_mode;
		input_tensor_ptr = output_tensor_ptrs[i % 2];
		cutf::memory::free_async(workspace, cuda_stream);
	}
}

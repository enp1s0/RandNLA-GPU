#include <cassert>
#include <cutf/cutensor.hpp>
#include <cutf/curand.hpp>
#include <cutf/cublas.hpp>
#include <cutf/error.hpp>
#include <cutf/memory.hpp>
#include <cuta/cutensor_utils.hpp>
#include "eval.hpp"

void mtk::rsvd_test::contract (
		cutensorHandle_t cutensor_handle,
		float* const out_ptr,
		const float* const core_tensor_ptr,
		const cuta::mode_t& core_tensor_mode,
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
					workspace_ptr, work_size, cuda_stream
					));

		current_output_mode = output_mode;
		input_tensor_ptr = output_tensor_ptrs[i % 2];
		cutf::memory::free_async(workspace, cuda_stream);
	}
}

namespace {
__global__ void rand_mp_kernel(
		float* const ptr,
		const std::size_t size
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= size) return;
	ptr[tid] = (ptr[tid] - 0.5) * 2;
}
void rand_mp(
		float* const ptr,
		const std::size_t size,
		cudaStream_t cuda_stream
		) {
	constexpr unsigned block_size = 256;
	rand_mp_kernel<<<(size + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(ptr, size);
}
void gen_q_matrix(
		float* const ptr,
		const unsigned m, const unsigned n,
		const std::string name,
		curandGenerator_t curand_gen,
		cudaStream_t cuda_stream
		) {
	const unsigned rank = std::min(m, n) - 10;

	auto a_ptr = cutf::memory::malloc_async<float>(m * rank, cuda_stream);
	auto b_ptr = cutf::memory::malloc_async<float>(rank * n, cuda_stream);

	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(curand_gen, a_ptr, m * rank));
	rand_mp(a_ptr, m * rank, cuda_stream);
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(curand_gen, b_ptr, rank * n));
	rand_mp(b_ptr, rank * n, cuda_stream);

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	CUTF_CHECK_ERROR(cublasSetStream(*cublas_handle.get(), cuda_stream));
	float alpah = 1.0f / rank, beta = 0.0f;
	CUTF_CHECK_ERROR(cutf::cublas::gemm(
				*cublas_handle.get(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, rank,
				&alpah,
				a_ptr, m,
				b_ptr, rank,
				&beta,
				ptr, m
				));
}
}

void mtk::rsvd_test::gen_input_tensor(
		cutensorHandle_t cutensor_handle,
		float* const out_ptr,
		float* const core_tensor_ptr,
		const cuta::mode_t& core_tensor_mode,
		const std::vector<float*> q_matrices,
		const std::vector<cuta::mode_t> q_matrix_modes,
		float* const work_ptr,
		const std::string tensor_name,
		cudaStream_t cuda_stream
		) {
	unsigned long long seed = 10;
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(curandSetStream(*cugen.get(), cuda_stream));

	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), core_tensor_ptr, cuta::utils::get_num_elements(core_tensor_mode)));
	for (unsigned i = 0; i < core_tensor_mode.size(); i++) {
		gen_q_matrix(q_matrices[i], q_matrix_modes[i][0].second, q_matrix_modes[i][1].second, tensor_name, *cugen.get(), cuda_stream);
	}
	mtk::rsvd_test::contract(
			cutensor_handle,
			out_ptr,
			core_tensor_ptr,
			core_tensor_mode,
			q_matrices,
			q_matrix_modes,
			work_ptr,
			cuda_stream
			);
}

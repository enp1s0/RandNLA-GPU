#ifndef __RANDNLA_TENSOR_EVAL__
#define __RANDNLA_TENSOR_EVAL__
#include <iostream>
#include <cuta/utils.hpp>
#include <cutf/cutensor.hpp>

namespace mtk {
namespace rsvd_test {
void contract (
		cutensorHandle_t cutensor_handle,
		float* const out_ptr,
		const float* const core_tensor_ptr,
		cuta::mode_t core_tensor_mode,
		const std::vector<float*> q_matrices,
		const std::vector<cuta::mode_t> q_matrix_modes,
		float* const work_ptr,
		cudaStream_t cuda_stream
		);
} // namespace rsvd_test
} // namespace mtk
#endif

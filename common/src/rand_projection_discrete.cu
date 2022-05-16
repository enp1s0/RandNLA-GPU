#include <vector>
#include <cutf/curand.hpp>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <rand_projection_base.hpp>
#include <curand_kernel.h>
#include "cuda_common.hpp"

namespace {
__global__ void rand_kernel(
		float* const dst_ptr,
		const std::size_t array_size,
		const float* const candidates_ptr,
		const float* const candidates_prob_ptr,
		const std::size_t candidates_size,
		const std::uint64_t seed
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= array_size) {
		return;
	}

	curandStateXORWOW_t curand_status;
	curand_init(seed, tid, 0, &curand_status);

	for (unsigned i = tid; i < array_size; i += gridDim.x * blockDim.x) {
		const auto r = static_cast<float>(curand(&curand_status)) / 0x100000000lu;
		std::size_t j = 0;
		for (; j < candidates_size - 1; j++) {
			if (r < candidates_prob_ptr[j]) {
				break;
			}
		}
		__syncwarp();

		dst_ptr[tid] = candidates_ptr[j];
	}
}
}

void mtk::rsvd_test::random_projection_discrete::gen_rand(const std::uint64_t seed) {
	std::vector<float> acc_rands;
	acc_rands.push_back(random_candidate_probs[0]);
	for (unsigned i = 1; i < random_candidate_probs.size(); i++) {
		acc_rands.push_back(acc_rands[i - 1] + random_candidate_probs[i]);
	}
	for (auto &p : acc_rands) {
		p /= acc_rands[acc_rands.size() - 1];
	}
	
	auto dev_rand_candidates      = cutf::memory::malloc_async<float>(acc_rands.size(), cuda_stream);
	auto dev_rand_candidate_probs = cutf::memory::malloc_async<float>(acc_rands.size(), cuda_stream);
	cutf::memory::copy_async(dev_rand_candidates, random_candidates.data(), acc_rands.size(), cuda_stream);
	cutf::memory::copy_async(dev_rand_candidate_probs, acc_rands.data()   , acc_rands.size(), cuda_stream);

	rand_kernel<<<256, 256, 0, cuda_stream>>>(
			rand_matrix_ptr, get_max_src_n() * get_max_target_rank(),
			dev_rand_candidates,
			dev_rand_candidate_probs,
			acc_rands.size(),
			seed
			);

	cutf::memory::free_async(dev_rand_candidate_probs, cuda_stream);
	cutf::memory::free_async(dev_rand_candidates, cuda_stream);
}

void mtk::rsvd_test::random_projection_discrete::apply(
		const std::size_t m, const std::size_t n, const std::size_t r,
		float* const dst_ptr, const std::size_t ldd,
		float* const src_ptr, const std::size_t lds
		) {
	const float alpha = 1.0f, beta = 0.0f;
	CUTF_CHECK_ERROR(cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				m, r, n,
				&alpha,
				src_ptr, lds,
				rand_matrix_ptr, r,
				&beta,
				dst_ptr, ldd
				));
}

void mtk::rsvd_test::random_projection_discrete::allocate_working_memory() {
	rand_matrix_ptr = cutf::memory::malloc_async<float>(get_max_src_n() * get_max_target_rank(), cuda_stream);
}

void mtk::rsvd_test::random_projection_discrete::free_working_memory() {
	cutf::memory::free_async(rand_matrix_ptr, cuda_stream);
}

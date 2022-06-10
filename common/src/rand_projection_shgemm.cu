#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/curand.hpp>
#include <curand_fp16/curand_fp16.hpp>
#include <rand_projection_base.hpp>
#include <mateval/minmax_cuda.hpp>
#include <cutf/debug/matrix.hpp>
#include "cuda_common.hpp"

namespace {
__global__ void float2half_kernel (
		half* const dst_ptr,
		const float* const src_ptr,
		const std::size_t size
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size) {
		return;
	}
	dst_ptr[tid] = cutf::type::cast<half>(src_ptr[tid]);
}

void float2half(
		half* const dst_ptr,
		const float* const src_ptr,
		const std::size_t size,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	float2half_kernel<<<(size + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
			dst_ptr,
			src_ptr,
			size
			);
}
} // unnamed namespace

void mtk::rsvd_test::random_projection_shgemm::gen_rand(const std::uint64_t seed) {
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(curandSetStream(*cugen.get(), cuda_stream));
	CUTF_CHECK_ERROR(cutf::curand::generate_normal(*cugen.get(), fp32_rand_matrix_ptr, get_max_src_n() * get_max_target_rank(), 0, 1));
	float2half(rand_matrix_ptr, fp32_rand_matrix_ptr, get_max_src_n() * get_max_target_rank(), cuda_stream);
}

void mtk::rsvd_test::random_projection_shgemm::apply(
		const std::size_t m, const std::size_t n, const std::size_t r,
		float* const dst_ptr, const std::size_t ldd,
		float* const src_ptr, const std::size_t lds
		) {
	const float alpha = 1.0f, beta = 0.0f;
	mtk::shgemm::shgemm(
				shgemm_handle,
				mtk::shgemm::op_n, mtk::shgemm::op_n,
				m, r, n,
				&alpha,
				src_ptr, lds,
				rand_matrix_ptr, n,
				&beta,
				dst_ptr, ldd,
				compute_type
				);
}

void mtk::rsvd_test::random_projection_shgemm::allocate_working_memory() {
	rand_matrix_ptr = cutf::memory::malloc_async<half>(get_max_src_n() * get_max_target_rank(), cuda_stream);
	fp32_rand_matrix_ptr = cutf::memory::malloc_async<float>(get_max_src_n() * get_max_target_rank(), cuda_stream);
}

void mtk::rsvd_test::random_projection_shgemm::free_working_memory() {
	cutf::memory::free_async(rand_matrix_ptr, cuda_stream);
}

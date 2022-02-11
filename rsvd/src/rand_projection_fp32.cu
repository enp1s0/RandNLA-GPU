#include <cutf/curand.hpp>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <rand_projection_base.hpp>

void mtk::rsvd_test::random_projection_fp32::gen_rand(const std::uint64_t seed) {
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(curandSetStream(*cugen.get(), cuda_stream));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), rand_matrix_ptr, get_src_n() * get_target_rank()));
}

void mtk::rsvd_test::random_projection_fp32::apply(
		float* const dst_ptr, const std::size_t ldd,
		float* const src_ptr, const std::size_t lds
		) {
	const float alpha = 1.0f, beta = 0.0f;
	CUTF_CHECK_ERROR(cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				get_src_m(), get_target_rank(), get_src_n(),
				&alpha,
				src_ptr, lds,
				rand_matrix_ptr, get_target_rank(),
				&beta,
				dst_ptr, ldd
				));
}

void mtk::rsvd_test::random_projection_fp32::allocate_working_memory() {
	rand_matrix_ptr = cutf::memory::malloc_async<float>(get_src_n() * get_target_rank(), cuda_stream);
}

void mtk::rsvd_test::random_projection_fp32::free_working_memory() {
	cutf::memory::free_async(rand_matrix_ptr, cuda_stream);
}

#include <cutf/curand.hpp>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <rand_projection_base.hpp>

void mtk::rsvd_test::random_projection_tf32::gen_rand(const std::uint64_t seed) {
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(curandSetStream(*cugen.get(), cuda_stream));
	CUTF_CHECK_ERROR(cutf::curand::generate_normal(*cugen.get(), rand_matrix_ptr, get_max_src_n() * get_max_target_rank(), 0, 1));
}

void mtk::rsvd_test::random_projection_tf32::apply(
		const std::size_t m, const std::size_t n, const std::size_t r,
		float* const dst_ptr, const std::size_t ldd,
		float* const src_ptr, const std::size_t lds
		) {
	cublasMath_t org_math_mode;
	CUTF_CHECK_ERROR(cublasGetMathMode(cublas_handle, &org_math_mode));

	CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
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

	CUTF_CHECK_ERROR(cublasSetMathMode(cublas_handle, org_math_mode));
}

void mtk::rsvd_test::random_projection_tf32::allocate_working_memory() {
	rand_matrix_ptr = cutf::memory::malloc_async<float>(get_max_src_n() * get_max_target_rank(), cuda_stream);
}

void mtk::rsvd_test::random_projection_tf32::free_working_memory() {
	cutf::memory::free_async(rand_matrix_ptr, cuda_stream);
}

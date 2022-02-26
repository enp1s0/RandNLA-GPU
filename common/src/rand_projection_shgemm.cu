#include <cutf/memory.hpp>
#include <curand_fp16/curand_fp16.hpp>
#include <rand_projection_base.hpp>

void mtk::rsvd_test::random_projection_shgemm::gen_rand(const std::uint64_t seed) {
	mtk::curand_fp16::generator_t gen;
	mtk::curand_fp16::create(gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	mtk::curand_fp16::set_seed(gen, seed);
	mtk::curand_fp16::set_cuda_stream(gen, cuda_stream);
	mtk::curand_fp16::uniform(gen, rand_matrix_ptr, get_max_src_n() * get_max_target_rank());
	mtk::curand_fp16::destroy(gen);
}

void mtk::rsvd_test::random_projection_shgemm::apply(
		const std::size_t m, const std::size_t n, const std::size_t r,
		float* const dst_ptr, const std::size_t ldd,
		float* const src_ptr, const std::size_t lds
		) {
	const float alpha = 1.0f, beta = 0.0f;
	mtk::shgemm::shgemm(
				shgemm_handle,
				mtk::shgemm::op_n, mtk::shgemm::op_t,
				m, r, n,
				&alpha,
				src_ptr, lds,
				rand_matrix_ptr, r,
				&beta,
				dst_ptr, ldd
				);
}

void mtk::rsvd_test::random_projection_shgemm::allocate_working_memory() {
	rand_matrix_ptr = cutf::memory::malloc_async<half>(get_max_src_n() * get_max_target_rank(), cuda_stream);
}

void mtk::rsvd_test::random_projection_shgemm::free_working_memory() {
	cutf::memory::free_async(rand_matrix_ptr, cuda_stream);
}

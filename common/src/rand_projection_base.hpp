#ifndef __MP_RSVD_RAND_PROJECTION_BASE_HPP__
#define __MP_RSVD_RAND_PROJECTION_BASE_HPP__
#include <string>
#include <shgemm/shgemm.hpp>
#include <cutf/cublas.hpp>

namespace mtk {
namespace rsvd_test {
class random_projection_base {
	const std::string name;
	std::size_t src_m, src_n, target_rank;
protected:
	cudaStream_t cuda_stream;
public:
	random_projection_base(
			const std::string name
			) :
		name(name), cuda_stream(0) {}

	void set_config(
			const std::size_t i_src_m,
			const std::size_t i_src_n,
			const std::size_t i_target_rank,
			cudaStream_t const stream = 0
			) {
		src_m = i_src_m;
		src_n = i_src_n;
		target_rank = i_target_rank;
		cuda_stream = stream;
	}

	std::string get_name() const {return name;}
	std::size_t get_src_m() const {return src_m;}
	std::size_t get_src_n() const {return src_n;}
	std::size_t get_target_rank() const {return target_rank;}

	virtual void allocate_working_memory() = 0;
	virtual void free_working_memory() = 0;
	virtual void gen_rand(const std::uint64_t seed) = 0;
	virtual void apply(
			float* const dst_ptr, const std::size_t ldd,
			float* const src_ptr, const std::size_t lds
			) = 0;
};

class random_projection_fp32 : public random_projection_base {
	const std::string name;

	float* rand_matrix_ptr;

	cublasHandle_t cublas_handle;
public:
	random_projection_fp32(cublasHandle_t const cublas_handle) : random_projection_base("rndprj_FP32"), cublas_handle(cublas_handle) {}

	void allocate_working_memory();
	void free_working_memory();
	void gen_rand(const std::uint64_t seed);
	void apply(
			float* const dst_ptr, const std::size_t ldd,
			float* const src_ptr, const std::size_t lds
			);
};

class random_projection_tf32 : public random_projection_base {
	const std::string name;

	float* rand_matrix_ptr;

	cublasHandle_t cublas_handle;
public:
	random_projection_tf32(cublasHandle_t const cublas_handle) : random_projection_base("rndprj_TF32"), cublas_handle(cublas_handle) {}

	void allocate_working_memory();
	void free_working_memory();
	void gen_rand(const std::uint64_t seed);
	void apply(
			float* const dst_ptr, const std::size_t ldd,
			float* const src_ptr, const std::size_t lds
			);
};

class random_projection_shgemm : public random_projection_base {
	const std::string name;

	half* rand_matrix_ptr;

	mtk::shgemm::shgemmHandle_t& shgemm_handle;
public:
	random_projection_shgemm(
			mtk::shgemm::shgemmHandle_t& shgemm_handle
			) : random_projection_base("rndprj_shgemm"), shgemm_handle(shgemm_handle) {}

	void allocate_working_memory();
	void free_working_memory();
	void gen_rand(const std::uint64_t seed);
	void apply(
			float* const dst_ptr, const std::size_t ldd,
			float* const src_ptr, const std::size_t lds
			);
};
} // namespace rsvd_test
} // namespace mtk
#endif

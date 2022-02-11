#ifndef __RSVD_TEST_HPP__
#define __RSVD_TEST_HPP__
#include <cstdint>
#include <cutf/cusolver.hpp>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <cutf/debug/time_breakdown.hpp>
#include <rand_projection_base.hpp>
#include <svd_base.hpp>

namespace mtk {
namespace rsvd_test {
class rsvd_base {
protected:
	const std::string name;

	const unsigned m;
	const unsigned n;

	float* A_ptr;
	const unsigned lda;

	const unsigned k;
	const unsigned p;

	float* U_ptr;
	const unsigned ldu;

	float* S_ptr;

	float* V_ptr;
	const unsigned ldv;

	const unsigned n_iter;

	cudaStream_t cuda_stream;

	cutf::debug::time_breakdown::profiler profiler;
protected:
	rsvd_base(
			const std::string name,
			const unsigned m, const unsigned n,
			const unsigned k, const unsigned p,
			const unsigned n_iter,
			float* const A_ptr, const unsigned lda,
			float* const U_ptr, const unsigned ldu,
			float* const S_ptr,
			float* const V_ptr, const unsigned ldv,
			cudaStream_t const cuda_stream
			):
		name(name),
		m(m), n(n),
		k(k), p(p),
		n_iter(n_iter),
		A_ptr(A_ptr), lda(lda),
		U_ptr(U_ptr), ldu(ldu),
		S_ptr(S_ptr),
		V_ptr(V_ptr), ldv(ldv),
		cuda_stream(cuda_stream),
		profiler(cuda_stream)	{}

public:
	virtual void prepare() = 0;
	virtual void run() = 0;
	virtual void clean() = 0;

	unsigned get_m() const {return m;}
	unsigned get_n() const {return n;}
	unsigned get_k() const {return k;}
	unsigned get_p() const {return p;}
	unsigned get_n_iter() const {return n_iter;}

	std::string get_name() const {
		return name;
	}

	void set_input_ptr(
			float* const A
			) {
		A_ptr = A;
	}
	void set_output_ptr(
			float* const U,
			float* const S,
			float* const V
			) {
		U_ptr = U;
		S_ptr = S;
		V_ptr = V;
	}

	void print_time_breakdown(const bool csv = false) const {
		if (csv) {
			profiler.print_result_csv();
		} else {
			profiler.print_result();
		}
	}
};

class rsvd_cusolver : public rsvd_base {
	cusolverDnHandle_t cusolver_handle;
	cusolverDnParams_t cusolver_params;

	// working memory size
	std::size_t working_memory_device_size;
	std::size_t working_memory_host_size;

	// working memory
	uint8_t* working_memory_device_ptr;
	uint8_t* working_memory_host_ptr;
	int* devInfo_ptr;
public:
	rsvd_cusolver(
			cusolverDnHandle_t cusolver_handle,
			cusolverDnParams_t cusolver_params,
			const unsigned m, const unsigned n,
			const unsigned k, const unsigned p,
			const unsigned n_iter,
			float* const A_ptr, const unsigned lda,
			float* const U_ptr, const unsigned ldu,
			float* const S_ptr,
			float* const V_ptr, const unsigned ldv,
			cudaStream_t const cuda_stream
			):
		cusolver_handle(cusolver_handle),
		cusolver_params(cusolver_params),
		rsvd_base("cusolver_svdr", m, n, k, p, n_iter, A_ptr, lda, U_ptr, ldu, S_ptr, V_ptr, ldv, cuda_stream) {}

	void prepare();
	void run();
	void clean();
};

class rsvd_selfmade : public rsvd_base {
	cusolverDnHandle_t cusolver_handle;
	cublasHandle_t cublas_handle;

	mtk::rsvd_test::random_projection_base& rand_proj;
	mtk::rsvd_test::svd_base& svd;

	// working memory size
	struct {
		float* alloc_ptr;
		std::size_t geqrf_0_size;
		float* geqrf_0_ptr;
		std::size_t orgqr_0_size;
		float* orgqr_0_ptr;
		std::size_t tau_size;
		float* tau_ptr;
		std::size_t gesvdj_size;
		float* gesvdj_ptr;
		std::size_t y_matrix_size;
		float* y_matrix_ptr;
		std::size_t b_matrix_size;
		float* b_matrix_ptr;
		std::size_t small_u_size;
		float* small_u_ptr;
		std::size_t full_V_size;
		float* full_V_ptr;
		std::size_t full_S_size;
		float* full_S_ptr;
		float* X_ptr;
		float* X_tmp_ptr[2];
		std::size_t X_tmp_size;
		int* devInfo_ptr;
	} working_memory;
public:
	rsvd_selfmade(
			cublasHandle_t cublas_handle,
			cusolverDnHandle_t cusolver_handle,
			cusolverDnParams_t cusolver_params,
			const unsigned m, const unsigned n,
			const unsigned k, const unsigned p,
			const unsigned n_iter,
			float* const A_ptr, const unsigned lda,
			float* const U_ptr, const unsigned ldu,
			float* const S_ptr,
			float* const V_ptr, const unsigned ldv,
			cudaStream_t const cuda_stream,
			mtk::rsvd_test::svd_base& svd,
			mtk::rsvd_test::random_projection_base& rand_proj
			):
		cublas_handle(cublas_handle),
		cusolver_handle(cusolver_handle),
		rsvd_base(
#ifdef FP16_EMULATION
				"fp16_emu",
#else
				std::string("selfmade-") + rand_proj.get_name() + "-" + svd.get_name_str(),
#endif
				m, n, k, p, n_iter, A_ptr, lda, U_ptr, ldu, S_ptr, V_ptr, ldv, cuda_stream),
				svd(svd), rand_proj(rand_proj)	{}

	void prepare();
	void run();
	void clean();
};


class svdj_cusolver : public rsvd_base {
	cusolverDnHandle_t cusolver_handle;
	gesvdjInfo_t svdj_params;

	// working memory size
	std::size_t working_memory_device_size;

	float* full_U_ptr;
	float* full_S_ptr;
	float* full_V_ptr;

	// working memory
	float* working_memory_device_ptr;
	int* devInfo_ptr;
public:
	svdj_cusolver(
			cusolverDnHandle_t cusolver_handle,
			const unsigned m, const unsigned n,
			const unsigned k, const unsigned p,
			const unsigned n_iter,
			float* const A_ptr, const unsigned lda,
			float* const U_ptr, const unsigned ldu,
			float* const S_ptr,
			float* const V_ptr, const unsigned ldv,
			cudaStream_t const cuda_stream
			):
		cusolver_handle(cusolver_handle),
		rsvd_base("svdj", m, n, k, p, n_iter, A_ptr, lda, U_ptr, ldu, S_ptr, V_ptr, ldv, cuda_stream) {}

	void prepare();
	void run();
	void clean();
};

void copy_matrix(
		const std::size_t m, const std::size_t n,
		float* const dst_ptr, const std::size_t ldd,
		const float* const src_ptr, const std::size_t lds,
		cudaStream_t cuda_stream
		);
void transpose_matrix(
		const std::size_t dst_m, const std::size_t dst_n,
		float* const dst_ptr, const std::size_t ldd,
		const float* const src_ptr, const std::size_t lds,
		cudaStream_t cuda_stream
		);
}
} // namespace mtk
#endif

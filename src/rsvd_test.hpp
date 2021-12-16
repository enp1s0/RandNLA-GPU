#ifndef __RSVD_TEST_HPP__
#define __RSVD_TEST_HPP__
#include <cstdint>
#include <cutf/cusolver.hpp>
#include <cutf/memory.hpp>

namespace mtk {
namespace rsvd_test {
class rsvd_base {
protected:
	const unsigned m;
	const unsigned n;

	float* A_ptr;
	const unsigned lda;

	const unsigned k;
	const unsigned p;

	float* U_ptr;
	const unsigned ldu;

	float* S_ptr;

	float* Vt_ptr;
	const unsigned ldvt;

	const unsigned n_svdj_iter;
protected:
	rsvd_base(
		const unsigned m, const unsigned n,
		const unsigned k, const unsigned p,
		const unsigned n_svdj_iter,
		float* const A_ptr, const unsigned lda,
		float* const U_ptr, const unsigned ldu,
		float* const S_ptr,
		float* const Vt_ptr, const unsigned ldvt
		):
		m(m), n(n),
		k(k), p(p),
		n_svdj_iter(n_svdj_iter),
		A_ptr(A_ptr), lda(lda),
		U_ptr(U_ptr), ldu(ldu),
		S_ptr(S_ptr),
		Vt_ptr(Vt_ptr), ldvt(ldvt) {}

public:
	virtual void prepare() = 0;
	virtual void run() = 0;

	unsigned get_m() const {return m;}
	unsigned get_n() const {return n;}
	unsigned get_k() const {return k;}
	unsigned get_p() const {return p;}

	void set_input_ptr(
			float* const A
			) {
		A_ptr = A;
	}
	void set_output_ptr(
			float* const U,
			float* const S,
			float* const Vt
			) {
		U_ptr = U;
		S_ptr = S;
		Vt_ptr = Vt;
	}
};

class rsvd_cusolver : public rsvd_base {
	cusolverDnHandle_t cusolver_handle;
	cusolverDnParams_t cusolver_params;

	// working memory size
	std::size_t working_memory_device_size;
	std::size_t working_memory_host_size;

	// working memory
	cutf::memory::device_unique_ptr<uint8_t> working_memory_device_uptr;
	cutf::memory::host_unique_ptr<uint8_t> working_memory_host_uptr;
	cutf::memory::device_unique_ptr<int> devInfo_uptr;
public:
	rsvd_cusolver(
		cusolverDnHandle_t cusolver_handle,
		cusolverDnParams_t cusolver_params,
		const unsigned m, const unsigned n,
		const unsigned k, const unsigned p,
		const unsigned n_svdj_iter,
		float* const A_ptr, const unsigned lda,
		float* const U_ptr, const unsigned ldu,
		float* const S_ptr,
		float* const Vt_ptr, const unsigned ldvt
		):
		cusolver_handle(cusolver_handle),
		cusolver_params(cusolver_params),
		rsvd_base(m, n, k, p, n_svdj_iter, A_ptr, lda, U_ptr, ldu, S_ptr, Vt_ptr, ldvt) {}

	void prepare();
	void run();
};
}
} // namespace mtk
#endif

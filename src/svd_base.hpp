#ifndef __MP_RSVD_SVD_BASE_HPP__
#define __MP_RSVD_SVD_BASE_HPP__
#include <cstdint>
#include <cusolver.h>

namespace mtk {
namespace rsvd_test {
struct svd_base {
protected:
	const std::size_t m, n;

	std::size_t work_size;

	cusolverDnHandle_t const cusolver_handle;
public:
	svd_base(
			const std::size_t m, const std::size_t n,
			cusolverDnHandle_t const cusolver_handle
			) : m(m), n(n), cusolver_handle(cusolver_handle) {}

	virtual void run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr) = 0;
	virtual std::size_t get_working_mem_size_in_byte() = 0;

	virtual char op_v() const = 0;
};

struct svd_qr : public svd_base {
public:
	svd_qr(
			const std::size_t m, const std::size_t n,
			cusolverDnHandle_t const cusolver_handle)
		: svd_base(m, n, cusolver_handle) {}

	void run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr);
	std::size_t get_working_mem_size_in_byte();

	char op_v() const {return 'T';}
};
} // rsvd_test
} // mtk
#endif

#ifndef __MP_RSVD_SVD_BASE_HPP__
#define __MP_RSVD_SVD_BASE_HPP__
#include <cstdint>
#include <string>
#include <cusolver.h>

namespace mtk {
namespace rsvd_test {
struct svd_base {
protected:
	const std::size_t m, n;

	std::size_t work_size;

	cusolverDnHandle_t const cusolver_handle;

	const std::string name;
public:
	svd_base(
			const std::string name,
			const std::size_t m, const std::size_t n,
			cusolverDnHandle_t const cusolver_handle
			) : name(name), m(m), n(n), cusolver_handle(cusolver_handle) {}

	std::string get_name_str() const {return name;};

	virtual void run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr) = 0;
	virtual std::size_t get_working_mem_size_in_byte() = 0;

	virtual char op_v() const = 0;

	virtual void preprocess();
};

struct svd_qr : public svd_base {
public:
	svd_qr(
			const std::size_t m, const std::size_t n,
			cusolverDnHandle_t const cusolver_handle)
		: svd_base("svd_qr", m, n, cusolver_handle) {}

	void run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr);
	std::size_t get_working_mem_size_in_byte();

	char op_v() const {return 'T';}
	void preprocess() {};
};

struct svd_jaccobi : public svd_base {
public:
	svd_jaccobi(
			const std::size_t m, const std::size_t n,
			cusolverDnHandle_t const cusolver_handle)
		: svd_base("svd_jaccobi", m, n, cusolver_handle) {}

	void run(
		float* const S_ptr,
		float* const U_ptr, const std::size_t ldu,
		float* const V_ptr, const std::size_t ldv,
		float* const input_ptr, const std::size_t ld,
		float* const work_ptr);
	std::size_t get_working_mem_size_in_byte();

	char op_v() const {return 'N';}
	void preprocess();
};
} // rsvd_test
} // mtk
#endif

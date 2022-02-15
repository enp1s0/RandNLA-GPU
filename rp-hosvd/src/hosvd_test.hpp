#ifndef __RANDNLA_GPU_HOSVD_HPP__
#define __RANDNLA_GPU_HOSVD_HPP__
#include <vector>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <rand_projection_base.hpp>
#include <cutf/debug/time_breakdown.hpp>
namespace mtk {
namespace rsvd_test {
class hosvd_base {
protected:
	std::vector<std::size_t> mode;
	std::vector<std::size_t> rank;

	float* const A_ptr;
	float* const S_ptr;
	std::vector<float*>& Q_ptr;

	const std::string name;

	cudaStream_t cuda_stream;
	cusolverDnHandle_t cusolver_handle;

	cutf::debug::time_breakdown::profiler profiler;
public:
	hosvd_base(
			const std::string name,
			const std::vector<std::size_t> mode,
			const std::vector<std::size_t> rank,
			float* const A_ptr,
			float* const S_ptr,
			std::vector<float*>& Q_ptr,
			cudaStream_t cuda_stream,
			cusolverDnHandle_t cusolver_handle
			) : name(name),
		mode(mode), rank(rank),
		cuda_stream(cuda_stream),
		A_ptr(A_ptr), S_ptr(S_ptr), Q_ptr(Q_ptr),
		profiler(cuda_stream),
		cusolver_handle(cusolver_handle)	{}

	std::string get_name_str() const {return std::string("hosvd-") + name;}

	virtual void prepare() = 0;
	virtual void run() = 0;
	virtual void clean() = 0;

	void print_time_breakdown(const bool csv = false) const {
		if (csv) {
			profiler.print_result_csv();
		} else {
			profiler.print_result();
		}
	}
};

class hosvd_rp : public hosvd_base {
	mtk::rsvd_test::random_projection_base& random_projection;

	struct {
		float* alloc_ptr;
		std::size_t alloc_size;
		float* ttgt_ptr;
		std::size_t ttgt_size;
		float* qr_ptr;
		float* tau_ptr;
		int* dev_ptr;
		std::size_t qr_size;
		std::vector<std::size_t> geqrf_size;
		std::vector<std::size_t> orgqr_size;
		std::size_t tau_size;
	} working_memory;
public:
	hosvd_rp(
			const std::vector<std::size_t> mode,
			const std::vector<std::size_t> rank,
			float* const A_ptr,
			float* const S_ptr,
			std::vector<float*>& Q_ptr,
			mtk::rsvd_test::random_projection_base& random_projection,
			cudaStream_t cuda_stream,
			cusolverDnHandle_t cusolver_handle
			) : 
		random_projection(random_projection),
		hosvd_base(std::string("rp-") + random_projection.get_name(), mode, rank, A_ptr, S_ptr, Q_ptr, cuda_stream, cusolver_handle) {}

	void prepare();
	void run();
	void clean();
};
} // namespace rsvd_test
} // namespace mtk
#endif

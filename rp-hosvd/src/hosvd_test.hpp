#ifndef __RANDNLA_GPU_HOSVD_HPP__
#define __RANDNLA_GPU_HOSVD_HPP__
#include <vector>
#include <cutensor.h>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <rand_projection_base.hpp>
#include <cutf/debug/time_breakdown.hpp>
#include <cutt/utils.hpp>
namespace mtk {
namespace rsvd_test {
class hosvd_base {
protected:
	const cutt::mode_t input_tensor_mode;
	const cutt::mode_t core_tensor_mode;

	float* const A_ptr;
	float* const S_ptr;
	std::vector<float*>& Q_ptr;

	const std::string name;

	cudaStream_t cuda_stream;
	cusolverDnHandle_t cusolver_handle;
	cutensorHandle_t cutensor_handle;

	cutf::debug::time_breakdown::profiler profiler;
public:
	hosvd_base(
			const std::string name,
			const cutt::mode_t input_tensor_mode,
			const cutt::mode_t core_tensor_mode,
			float* const A_ptr,
			float* const S_ptr,
			std::vector<float*>& Q_ptr,
			cudaStream_t cuda_stream,
			cusolverDnHandle_t cusolver_handle,
			cutensorHandle_t cutensor_handle
			) : name(name),
		input_tensor_mode(input_tensor_mode),
		core_tensor_mode(core_tensor_mode),
		cuda_stream(cuda_stream),
		A_ptr(A_ptr), S_ptr(S_ptr), Q_ptr(Q_ptr),
		profiler(cuda_stream),
		cusolver_handle(cusolver_handle),
		cutensor_handle(cutensor_handle)	{}

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

	std::vector<cutt::mode_t> Q_tensor_mode;
	std::vector<uint32_t> Q_tensor_alignment_requirement;
	std::vector<cutensorTensorDescriptor_t> Q_tensor_desc;
	std::vector<cutt::mode_t> tmp_core_tensor_mode;
	std::vector<uint32_t> tmp_core_tensor_alignment_requirement;
	std::vector<cutensorTensorDescriptor_t> tmp_core_tensor_desc;

	std::vector<cutensorContractionDescriptor_t> contraction_desc;
	std::vector<cutensorContractionFind_t> contraction_find;
	std::vector<std::size_t> contraction_working_mem_size;
	std::vector<cutensorContractionPlan_t> contraction_plan;
	std::size_t contraction_working_mem_size_max;
	void* contraction_working_mem_ptr;
public:
	hosvd_rp(
			const cutt::mode_t input_tensor_mode,
			const cutt::mode_t core_tensor_mode,
			float* const A_ptr,
			float* const S_ptr,
			std::vector<float*>& Q_ptr,
			mtk::rsvd_test::random_projection_base& random_projection,
			cudaStream_t cuda_stream,
			cusolverDnHandle_t cusolver_handle,
			cutensorHandle_t cutensor_handle
			) : 
		random_projection(random_projection),
		hosvd_base(std::string("rp-") + random_projection.get_name(), input_tensor_mode, core_tensor_mode, A_ptr, S_ptr, Q_ptr, cuda_stream, cusolver_handle, cutensor_handle) {}

	void prepare();
	void run();
	void clean();
};
} // namespace rsvd_test
} // namespace mtk
#endif

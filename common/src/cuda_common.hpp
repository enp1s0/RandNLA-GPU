#ifndef __RANDNLA_CUDA_COMMON_HPP__
#define __RANDNLA_CUDA_COMMON_HPP__
#include <cstdint>
namespace mtk {
namespace rsvd_test {
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
} // namespace rsvd_test
} // namespace mtk
#endif

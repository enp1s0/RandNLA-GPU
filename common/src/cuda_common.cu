#include "cuda_common.hpp"
namespace {
__global__ void copy_kernel(
		const std::size_t m, const std::size_t n,
		float* const dst_ptr, const std::size_t ldd,
		const float* const src_ptr, const std::size_t lds
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto tm = tid % m;
	const auto tn = tid / m;

	const auto dst_index = tm + tn * ldd;
	const auto src_index = tm + tn * lds;

	dst_ptr[dst_index] = src_ptr[src_index];
}

__global__ void transpose_kernel(
		const std::size_t dst_m, const std::size_t dst_n,
		float* const dst_ptr, const std::size_t ldd,
		const float* const src_ptr, const std::size_t lds
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= dst_m * dst_n) {
		return;
	}

	const auto tm = tid % dst_m;
	const auto tn = tid / dst_m;

	const auto dst_index = tm + tn * ldd;
	const auto src_index = tn + tm * lds;

	dst_ptr[dst_index] = src_ptr[src_index];
}
} // noname namespace

void mtk::rsvd_test::copy_matrix(
		const std::size_t m, const std::size_t n,
		float* const dst_ptr, const std::size_t ldd,
		const float* const src_ptr, const std::size_t lds,
		cudaStream_t cuda_stream
		) {
	const auto num_threads = m * n;
	const unsigned block_size = 256;
	const unsigned grid_size = (num_threads + block_size - 1) / block_size;

	copy_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
			m, n,
			dst_ptr, ldd,
			src_ptr, lds
			);
}

void mtk::rsvd_test::transpose_matrix(
		const std::size_t dst_m, const std::size_t dst_n,
		float* const dst_ptr, const std::size_t ldd,
		const float* const src_ptr, const std::size_t lds,
		cudaStream_t cuda_stream
		) {
	const auto num_threads = dst_m * dst_n;
	const unsigned block_size = 256;
	const unsigned grid_size = (num_threads + block_size - 1) / block_size;

	transpose_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
			dst_m, dst_n,
			dst_ptr, ldd,
			src_ptr, lds
			);
}

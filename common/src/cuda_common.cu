#include "cuda_common.hpp"
#include <cutf/type.hpp>
#include <cutf/experimental/fp.hpp>
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

template <class T>
__global__ void cut_mantissa_kernel(
		T* const ptr,
		const std::size_t size,
		const std::size_t remain_mantissa_length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size) {
		return;
	}

	auto v = cutf::experimental::fp::reinterpret_as_fp<T>(
			cutf::experimental::fp::reinterpret_as_uint<typename cutf::experimental::fp::same_size_fp<T>::type>(ptr[tid]) & (~((1u << (cutf::experimental::fp::get_mantissa_size<T>() - remain_mantissa_length)) - 1))
			);
	if (cutf::type::cast<float>(v) < 1.f / (1u << 25)) {
		v = 0;
	}
	ptr[tid] = v;
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

void mtk::rsvd_test::cut_mantissa(
		half* const ptr,
		const std::size_t size,
		const std::size_t remain_mantissa_length,
		cudaStream_t cuda_stream
		) {
	const unsigned block_size = 256;
	cut_mantissa_kernel<half><<<(size + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
			ptr, size, remain_mantissa_length
			);
}
void mtk::rsvd_test::cut_mantissa(
		float* const ptr,
		const std::size_t size,
		const std::size_t remain_mantissa_length,
		cudaStream_t cuda_stream
		) {
	const unsigned block_size = 256;
	cut_mantissa_kernel<float><<<(size + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
			ptr, size, remain_mantissa_length
			);
}

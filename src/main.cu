#include <iostream>
#include <rsvd_test.hpp>
#include <cutf/memory.hpp>
#include <cutf/cusolver.hpp>

constexpr unsigned max_log_m = 15;
constexpr unsigned max_log_n = 15;
constexpr unsigned n_svdj_iter = 10;

namespace {
void evaluate(
		const std::string test_name,
		mtk::rsvd_test::rsvd_base& rsvd,
		const unsigned n_tests
		) {
	std::printf("%s,%u,%u,%u,%u,%u,",
			test_name.c_str(),
			rsvd.get_m(),
			rsvd.get_n(),
			rsvd.get_k(),
			rsvd.get_p(),
			rsvd.get_n_svdj_iter()
			);
	const auto A_size = rsvd.get_m() * rsvd.get_n();
	const auto S_size = std::min(rsvd.get_m(), rsvd.get_n());
	const auto U_size = rsvd.get_m() * (rsvd.get_k() + rsvd.get_p());
	const auto V_size = rsvd.get_n() * (rsvd.get_k() + rsvd.get_p());

	auto A_uptr = cutf::memory::get_device_unique_ptr<float>(A_size);
	auto U_uptr = cutf::memory::get_device_unique_ptr<float>(U_size);
	auto S_uptr = cutf::memory::get_device_unique_ptr<float>(S_size);
	auto V_uptr = cutf::memory::get_device_unique_ptr<float>(V_size);

	rsvd.set_input_ptr(A_uptr.get());
	rsvd.set_output_ptr(U_uptr.get(), S_uptr.get(), V_uptr.get());

	rsvd.prepare();

	rsvd.run();

	std::printf("%u\n", n_tests);
}
} // noname namespace

int main() {
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

	for (unsigned log_m = 5; log_m <= max_log_m; log_m++) {
		for (unsigned log_n = 5; log_n <= max_log_n; log_n++) {
			const auto max_log_k = std::min(log_m, log_n);
			for (unsigned log_k = 4; log_k <= max_log_k - 1; log_k++) {
				const auto m = 1u << log_m;
				const auto n = 1u << log_n;
				const auto k = 1u << log_k;
				const auto p = k / 10;
				if (k + p > std::min(m, n)) {
					break;
				}

				mtk::rsvd_test::rsvd_cusolver rsvd_cusolver(
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, k, p, n_svdj_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, n
						);

				evaluate("cusolver", rsvd_cusolver, 10);
			}
		}
	}
}

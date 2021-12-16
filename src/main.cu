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
	const auto A_size = rsvd.get_m() * rsvd.get_n();
	const auto S_size = rsvd.get_k() + rsvd.get_p();
	const auto U_size = rsvd.get_m() * (rsvd.get_k() + rsvd.get_p());
	const auto Vt_size = rsvd.get_n() * (rsvd.get_k() + rsvd.get_p());

	auto A_uptr = cutf::memory::get_device_unique_ptr<float>(A_size);
	auto U_uptr = cutf::memory::get_device_unique_ptr<float>(U_size);
	auto S_uptr = cutf::memory::get_device_unique_ptr<float>(S_size);
	auto Vt_uptr = cutf::memory::get_device_unique_ptr<float>(Vt_size);

	rsvd.set_input_ptr(A_uptr.get());
	rsvd.set_output_ptr(U_uptr.get(), S_uptr.get(), Vt_uptr.get());

	rsvd.prepare();

	rsvd.run();
}
} // noname namespace

int main() {
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
	CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

	for (unsigned log_m = 5; log_m <= max_log_m; log_m++) {
		for (unsigned log_n = 5; log_n <= max_log_n; log_n++) {
			const auto max_log_k = std::min(log_m, log_n);
			for (unsigned log_k = 5; max_log_k; log_k++) {
				const auto m = 1u << log_m;
				const auto n = 1u << log_n;
				const auto k = 1u << log_k;
				const auto p = k / 10;

				mtk::rsvd_test::rsvd_cusolver rsvd_cusolver(
						*cusolver_handle.get(),
						*cusolver_params.get(),
						m, n, k, p, n_svdj_iter,
						nullptr, m,
						nullptr, m,
						nullptr,
						nullptr, k + p
						);

				evaluate("cusolver", rsvd_cusolver, 10);
			}
		}
	}
}

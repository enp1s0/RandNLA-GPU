#include <iostream>
#include <fstream>

enum date_state_t {
	none,
	breakdown_csv,
	breakdown_human,
	main_data
};

int main(int argc, char** argv) {
	date_state_t data_state = date_state_t::none;

	if (argc <= 1) {
		std::printf("Usage: %s [dst_dir]\n", argv[0]);
		return 1;
	}

	const std::string dst_dir = argv[1];

	const std::string sep_csv_start = "# START csv";
	const std::string sep_csv_end = "# END csv";
	const std::string sep_human_start = "# START human";
	const std::string sep_human_end = "# END human";
	const std::string sep_main_data = "implementation,";

	std::ofstream ofs;

	std::string line;
	while (std::getline(std::cin, line)) {
		if (line.find(sep_csv_start) != std::string::npos) {
			data_state = date_state_t::breakdown_csv;
			ofs.open(dst_dir + "/" + line.substr(sep_csv_start.length() + 1) + ".csv");
			continue;
		} else if (line.find(sep_csv_end) != std::string::npos) {
			data_state = date_state_t::main_data;
			ofs.close();
			continue;
		} else if (line.find(sep_human_start) != std::string::npos) {
			data_state = date_state_t::breakdown_human;
			ofs.open(dst_dir + "/" + line.substr(sep_human_start.length() + 1) + ".txt");
			continue;
		} else if (line.find(sep_human_end) != std::string::npos) {
			data_state = date_state_t::main_data;
			ofs.close();
			continue;
		} else if (line.find(sep_main_data) != std::string::npos) {
			data_state = date_state_t::main_data;
		}

		switch(data_state) {
			case date_state_t::main_data:
				std::cout << line << std::endl;
				break;
			case date_state_t::breakdown_csv:
			case date_state_t::breakdown_human:
				ofs << line << std::endl;
				break;
			default:
				break;
		}
	}
}

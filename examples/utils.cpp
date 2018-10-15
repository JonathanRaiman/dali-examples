#include "utils.h"

#include <dali/array/memory/device.h>
#include <dali/runtime_config.h>
#include <dali/utils/assert2.h>
#include <dali/utils/print_utils.h>

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port,      6379,        "Default port to be used by visualizer.");
DEFINE_string(visualizer,          "",          "What to name the visualization job.");
DEFINE_int32(device,              -1,           "What device to run the computation on (-1 for cpu, or number of gpu: 0, 1, ..).");

using std::string;
using std::stringstream;
using std::vector;

namespace utils {

    void map_to_file(const std::unordered_map<string, std::vector<string>>& map, const string& fname) {
        std::ofstream fp;
        fp.open(fname.c_str(), std::ios::out);
        for (auto& kv : map) {
            fp << kv.first;
            for (auto& v: kv.second)
                fp << " " << v;
            fp << "\n";
        }
    }

    std::unordered_map<string, std::vector<string>> text_to_map(const string& fname) {
            std::ifstream infile(fname);
            string line;
            const char space = ' ';
            std::unordered_map<string, std::vector<string>> map;
            while (std::getline(infile, line)) {
                    if (*line.begin() != '=' && *line.begin() != '-' && *line.begin() != '#') {
                            const auto tokens = utils::split(line, space);
                            if (tokens.size() > 1) {
                                    auto ptr = tokens.begin() + 1;
                                    while( ptr != tokens.end()) {
                                            map[tokens[0]].emplace_back(*(ptr++));
                                    }
                            }
                    }
            }
            return map;
    }

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights) {
        T minimum = weights[0];
        T sum = 0;
        for (int i=0; i<weights.size(); ++i) {
            minimum = std::min(minimum, weights[i]);
            sum += weights[i];
        }
        vector<T> res;
        T normalized_sum = sum - minimum * weights.size();
        for (int i=0; i<weights.size(); ++i) {
            res.push_back((weights[i] - minimum) / (normalized_sum));
        }
        return res;
    }

    template vector<float> normalize_weights(const std::vector<float>&);
    template vector<double> normalize_weights(const std::vector<double>&);

    template<typename T>
    void stream_to_list(T& fp, vector<string>& list) {
            string line;
            while (std::getline(fp, line))
                    list.emplace_back(line);
    }

    vector<string> load_list(const string& fname) {
        vector<string> list;
        if (is_gzip(fname)) {
            igzstream fpgz(fname.c_str(), std::ios::in | std::ios::binary);
            stream_to_list(fpgz, list);
        } else {
            std::fstream fp(fname, std::ios::in | std::ios::binary);
            stream_to_list(fp, list);
        }
        return list;
    }

    void update_device(int device_name) {
        if (device_name == -1) {
            memory::default_preferred_device = memory::Device::cpu();
        } else if (device_name >= 0) {
            #ifdef DALI_USE_CUDA
                memory::default_preferred_device = memory::Device::gpu(device_name);
            #else
                utils::assert2(
                    device_name >= -1,
                    utils::MS() << "Dali compiled without GPU support: "
                                << "update_device's device_name argument must be -1 for cpu"
                );

            #endif
        } else {
            utils::assert2(
                device_name >= -1,
                utils::MS() << "update_device's device_name argument must be >= -1"
            );
        }
    }


    template<typename T>
    void assert_map_has_key(std::unordered_map<string, T>& map, const string& key) {
            if (map.count(key) < 1) {
                    stringstream error_msg;
                    error_msg << "Map is missing the following key : \"" << key << "\".";
                    throw std::runtime_error(error_msg.str());
            }
    }

    template void assert_map_has_key(std::unordered_map<string, string>&, const string&);
    template void assert_map_has_key(std::unordered_map<string, vector<string>>&, const string&);
}

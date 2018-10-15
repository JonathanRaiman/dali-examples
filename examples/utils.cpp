#include "utils.h"
#include <ostream>
#include <fstream>
#include <dali/array/memory/device.h>
#include <dali/utils/assert2.h>
#include <dali/utils/print_utils.h>

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port,      6379,        "Default port to be used by visualizer.");
DEFINE_string(visualizer,          "",          "What to name the visualization job.");
DEFINE_int32(device,              -1,           "What device to run the computation on (-1 for cpu, or number of gpu: 0, 1, ..).");

namespace utils {

    void map_to_file(const std::unordered_map<std::string, std::vector<std::string>>& map, const std::string& fname) {
        std::ofstream fp;
        fp.open(fname.c_str(), std::ios::out);
        for (auto& kv : map) {
            fp << kv.first;
            for (auto& v: kv.second)
                fp << " " << v;
            fp << "\n";
        }
    }

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights) {
        T minimum = weights[0];
        T sum = 0;
        for (int i = 0; i < weights.size(); ++i) {
            minimum = std::min(minimum, weights[i]);
            sum += weights[i];
        }
        std::vector<T> res;
        T normalized_sum = sum - minimum * weights.size();
        for (int i = 0; i < weights.size(); ++i) {
            res.push_back((weights[i] - minimum) / (normalized_sum));
        }
        return res;
    }

    template std::vector<float> normalize_weights(const std::vector<float>&);
    template std::vector<double> normalize_weights(const std::vector<double>&);

    template<typename T>
    void stream_to_list(T& fp, std::vector<std::string>& list) {
            std::string line;
            while (std::getline(fp, line)) list.emplace_back(line);
    }

    void update_device(int device_name) {
        if (device_name == -1) {
            default_preferred_device = Device::cpu();
        } else if (device_name >= 0) {
            #ifdef DALI_USE_CUDA
                default_preferred_device = Device::gpu(device_name);
            #else
                ASSERT2(device_name >= -1, "Dali compiled without GPU support: update_device's device_name argument must be -1 for cpu");
            #endif
        } else {
            ASSERT2(device_name >= -1, "update_device's device_name argument must be >= -1");
        }
    }
}

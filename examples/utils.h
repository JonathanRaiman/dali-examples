#ifndef DALI_EXAMPLES_EXAMPLES_UTILS_H
#define DALI_EXAMPLES_EXAMPLES_UTILS_H

#include <dali/utils/core_utils.h>
#include <dali/utils/gzstream.h>
#include <gflags/gflags.h>
#include <string>
#include <vector>

DECLARE_string(visualizer_hostname);
DECLARE_int32(visualizer_port);
DECLARE_string(visualizer);
DECLARE_int32(device);

namespace utils {
    std::vector<std::string> load_list(const std::string&);

    void map_to_file(const std::unordered_map<std::string, str_sequence>&, const std::string&);


    /**
    Text To Map
    -----------
    Read a text file, extract all key value pairs and
    ignore markdown decoration characters such as =, -,
    and #

    Inputs
    ------
    std::string fname : the file to read

    Outputs
    -------
    std::unordered_map<string, std::vector<string> > map : the extracted key value pairs.

    **/
    std::unordered_map<std::string, str_sequence> text_to_map(const std::string&);

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights);

    // what device to run computation on
    // -1 for cpu, else indicates which gpu to use (will fail on cpu-only Dali)
    void update_device(int);

    template<typename T>
    void assert_map_has_key(std::unordered_map<std::string, T>&, const std::string&);
}




#endif

#ifndef DALI_EXAMPLES_EXAMPLES_UTILS_H
#define DALI_EXAMPLES_EXAMPLES_UTILS_H

#include <dali/utils/core_utils.h>
#include <dali/array/expression/expression.h>
#include <gflags/gflags.h>
#include <string>
#include <vector>
#include <unordered_set>
#include "config.h"

#define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
#define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

DECLARE_string(visualizer_hostname);
DECLARE_int32(visualizer_port);
DECLARE_string(visualizer);
DECLARE_int32(device);

namespace utils {
    void map_to_file(const std::unordered_map<std::string, str_sequence>&, const std::string&);

    template<typename T>
    std::vector<T> normalize_weights(const std::vector<T>& weights);

    // what device to run computation on
    // -1 for cpu, else indicates which gpu to use (will fail on cpu-only Dali)
    void update_device(int);
    void draw_expression_ownership_graph(const std::string& fname, const std::unordered_set<const Expression*>& expressions);
}

#endif

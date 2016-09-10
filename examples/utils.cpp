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

    template<typename T>
    void load_corpus_from_stream(Corpus& corpus, T& stream) {
        corpus.ParseFromIstream(&stream);
    }

    template void load_corpus_from_stream(Corpus&, igzstream&);
    template void load_corpus_from_stream(Corpus&, std::fstream&);
    template void load_corpus_from_stream(Corpus&, std::stringstream&);
    template void load_corpus_from_stream(Corpus&, std::istream&);

    Corpus load_corpus_protobuff(const std::string& path) {
        Corpus corpus;
        if (is_gzip(path)) {
            igzstream fpgz(path.c_str(), std::ios::in | std::ios::binary);
            load_corpus_from_stream(corpus, fpgz);
        } else {
            std::fstream fp(path, std::ios::in | std::ios::binary);
            load_corpus_from_stream(corpus, fp);
        }
        return corpus;
    }

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


    tokenized_labeled_dataset load_protobuff_dataset(string directory, const vector<string>& index2label) {
        ensure_directory(directory);
        auto files = listdir(directory);
        tokenized_labeled_dataset dataset;
        for (auto& file : files) {
            auto corpus = load_corpus_protobuff(directory + file);
            for (auto& example : corpus.example()) {
                dataset.emplace_back(std::initializer_list<vector<string>>({
                    vector<string>(example.words().begin(), example.words().end()),
                    triggers_to_strings(example.trigger(), index2label)
                }));
            }
        }
        return dataset;
    }


    tokenized_labeled_dataset load_protobuff_dataset(
        SQLite::Statement& query,
        const vector<string>& index2label,
        int num_elements,
        int column) {
        int els_seen = 0;
        tokenized_labeled_dataset dataset;
        while (query.executeStep()) {
            const char* protobuff_serialized = query.getColumn(column);
            stringstream ss(protobuff_serialized);
            Corpus corpus;
            load_corpus_from_stream(corpus, ss);
            for (auto& example : corpus.example()) {
                dataset.emplace_back(std::initializer_list<vector<string>>{
                    vector<string>(example.words().begin(), example.words().end()),
                    triggers_to_strings(example.trigger(), index2label)
                });
                ++els_seen;
            }
            if (els_seen >= num_elements) {
                break;
            }
        }
        return dataset;
    }

    str_sequence triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>& triggers, const str_sequence& index2target) {
        str_sequence data;
        data.reserve(triggers.size());
        for (auto& trig : triggers)
            if (trig.id() < index2target.size())
                data.emplace_back(index2target[trig.id()]);
        return data;
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

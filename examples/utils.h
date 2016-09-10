#ifndef DALI_EXAMPLES_EXAMPLES_UTILS_H
#define DALI_EXAMPLES_EXAMPLES_UTILS_H

#include <dali/utils/core_utils.h>
#include <dali/utils/gzstream.h>
#include <gflags/gflags.h>
#include <string>
#include <vector>

#include "SQLiteCpp/Database.h"
#include "protobuf/corpus.pb.h"

DECLARE_string(visualizer_hostname);
DECLARE_int32(visualizer_port);
DECLARE_string(visualizer);
DECLARE_int32(device);

namespace utils {
    template<typename T>
    void load_corpus_from_stream(Corpus& corpus, T& stream);

    Corpus load_corpus_protobuff(const std::string&);

    /**
    Load Protobuff Dataset
    ----------------------

    Load a set of protocol buffer serialized files from ordinary
    or gzipped files, and conver their labels from an index
    to their string representation using an index2label mapping.

    Inputs
    ------

    std::string directory : where the protocol buffer files are stored
    const std::vector<std::string>& index2label : mapping from numericals to
                                                  string labels

    Outputs
    -------

    utils::tokenized_multilabeled_dataset dataset : pairs of tokenized strings
                                                    and vector of string labels

    **/
    tokenized_labeled_dataset load_protobuff_dataset(std::string, const std::vector<std::string>&);

    tokenized_labeled_dataset load_protobuff_dataset(
            SQLite::Statement& query, const
            std::vector<std::string>&, int max_elements = 100, int column = 0);

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

    /**
    Triggers To Strings
    -------------------

    Convert triggers from an example to their
    string representation using an index2label
    string vector.

    Inputs
    ------

    const google::protobuf::RepeatedPtrField<Example::Trigger>& triggers : list of Trigger protobuff objects
    const vector<string>& index2target : mapping from numerical to string representation

    Outputs
    -------

    std::vector<std::string> data : the strings corresponding to the trigger targets
    **/
    std::vector<std::string> triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>&, const std::vector<std::string>&);

    // what device to run computation on
    // -1 for cpu, else indicates which gpu to use (will fail on cpu-only Dali)
    void update_device(int);


    template<typename T>
    void assert_map_has_key(std::unordered_map<std::string, T>&, const std::string&);
}




#endif

#include <algorithm>
#include <atomic>
#include <fstream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <chrono>

#include <dali/core.h>
#include <dali/data_processing/SST.h>
#include <dali/data_processing/Glove.h>
#include <dali_visualizer/visualizer.h>

#include "models/shallow_copy.h"
#include "models/standard_flags.h"
#include "models/stacked_model_builder.h"
#include "models/StackedGatedModel.h"
#include "utils.h"

using std::atomic;
using std::chrono::seconds;
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::stringstream;
using std::string;
using std::ofstream;
using std::min;
using utils::Vocab;
using utils::assert2;
using namespace dali::visualizer;

typedef double REAL_t;

DEFINE_int32(minibatch,           100,        "What size should be used for the minibatches ?");
DEFINE_int32(patience,            5,          "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_double(dropout,            0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(reg,                0.0,        "What penalty to place on L2 norm of weights?");
DEFINE_bool(fast_dropout,         true,       "Use fast dropout?");
DEFINE_string(test,               "",         "Where is the test set?");
DEFINE_double(root_weight,        1.0,        "By how much to weigh the roots in the objective function?");
DEFINE_string(pretrained_vectors, "",         "Load pretrained word vectors?");
DEFINE_string(results_file,       "",         "Where to save test performance.");
DEFINE_string(save_location,      "",         "Where to save test performance.");
DEFINE_int32(validation_metric,   0,          "Use root (1) or overall (0) objective to choose best validation parameters?");
DEFINE_double(embedding_learning_rate, -1.0,  "A separate learning rate for embedding layer");
DEFINE_bool(svd_init,             false,       "Initialize weights using SVD?");
DEFINE_bool(average_gradient,     false,      "Error during minibatch should be average or sum of errors.");
DEFINE_string(memory_penalty_curve, "flat",   "Type of annealing used on gate memory penalty (flat, linear, square)");

ThreadPool* pool;

int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Sentiment Analysis using single LSTM\n"
        "------------------------------------\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date April 7th 2015"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_patience > 40) FLAGS_patience = 40;

    int memory_penalty_curve_type;
    if (FLAGS_memory_penalty_curve == "flat") {
        memory_penalty_curve_type = 0;
    } else if (FLAGS_memory_penalty_curve == "linear") {
        memory_penalty_curve_type = 1;
    } else if (FLAGS_memory_penalty_curve == "square") {
        memory_penalty_curve_type = 2;
    } else {
        utils::assert2(false, "memory_penalty_curve can only be flat, linear, or square.");
    }

    auto epochs = FLAGS_epochs;
    int rampup_time = 10;

    auto sentiment_treebank = SST::load(FLAGS_train);
    auto embedding          = Mat<REAL_t>(100, 0);
    auto word_vocab         = Vocab();
    if (!FLAGS_pretrained_vectors.empty())
        glove::load(FLAGS_pretrained_vectors, &embedding, &word_vocab, 50000);
    else
        word_vocab = SST::get_vocabulary(sentiment_treebank, FLAGS_min_occurence);
    auto vocab_size     = word_vocab.size();
    auto dataset        = SST::convert_trees_to_indexed_minibatches(
        word_vocab,
        sentiment_treebank,
        FLAGS_minibatch
    );
    auto validation_set = SST::convert_trees_to_indexed_minibatches(
        word_vocab,
        SST::load(FLAGS_validation),
        FLAGS_minibatch
    );

    pool = new ThreadPool(FLAGS_j);
    // Create a model with an embedding, and several stacks:

    auto stack_size  = std::max(FLAGS_stack_size, 1);

    auto model = FLAGS_load.empty() ? StackedGatedModel<REAL_t>(
        FLAGS_pretrained_vectors.empty() ? word_vocab.size() : 0,
        FLAGS_pretrained_vectors.empty() ? FLAGS_hidden : embedding.dims(1),
        FLAGS_hidden,
        stack_size,
        SST::label_names.size(),
        (FLAGS_shortcut && stack_size > 1) ? FLAGS_shortcut : false,
        FLAGS_memory_feeds_gates,
        FLAGS_memory_penalty) : StackedGatedModel<REAL_t>::load(FLAGS_load);

    if (FLAGS_shortcut && stack_size == 1) {
        std::cout << "shortcut flag ignored: Shortcut connections only take effect with stack size > 1" << std::endl;
    }

    // don't send the input vector to the
    // decoder:
    model.input_vector_to_decoder(false);

    std::cout << "model.input_vector_to_decoder() = " << model.input_vector_to_decoder() << std::endl;

    std::cout << " Unique Trees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << "        Dropout Prob : " << FLAGS_dropout << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "   First Hidden Size : " << model.hidden_sizes[0] << std::endl
              << "           LSTM type : " << (model.memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << model.hidden_sizes.size() << std::endl
              << " # training examples : " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl
              << "     validation obj. : " << (FLAGS_validation_metric == 0 ? "overall" : "root") << std::endl
              << " # layers -> decoder : " << model.decoder.matrices.size() << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;
    if (FLAGS_embedding_learning_rate > 0)
        std::cout << " Embedding step size : " << FLAGS_embedding_learning_rate << std::endl;

    if (!FLAGS_pretrained_vectors.empty()) {
        std::cout << "  Pretrained Vectors : " << FLAGS_pretrained_vectors << std::endl;
        model.embedding = embedding;
    }

    vector<vector<Mat<REAL_t>>>       thread_params;
    vector<vector<Mat<REAL_t>>>       thread_embedding_params;
    vector<StackedGatedModel<REAL_t>> thread_models;
    std::tie(thread_models, thread_embedding_params, thread_params) = utils::shallow_copy_multi_params(model, FLAGS_j, [&model](const Mat<REAL_t>& mat) {
        return &mat.w().memory() == &model.embedding.w().memory();
    });

    vector<Mat<REAL_t>> params = model.parameters();
    vector<Mat<REAL_t>> embedding_params(params.begin(), params.begin() + 1);
    params = vector<Mat<REAL_t>>(params.begin() + 1, params.end());

    if (FLAGS_svd_init) {
        auto svd_init = weights<REAL_t>::svd(weights<REAL_t>::gaussian(0.0, 1.0));
        for (auto& param : params) {
            svd_init(param.w());
        }
    }

    auto solver           = Solver::construct(FLAGS_solver, params, (REAL_t) FLAGS_learning_rate, (REAL_t) FLAGS_reg);
    auto embedding_solver = Solver::construct(FLAGS_solver, embedding_params, (REAL_t) (FLAGS_embedding_learning_rate > 0 ? FLAGS_embedding_learning_rate : FLAGS_learning_rate), 0.0);

    std::tuple<REAL_t, REAL_t> best_validation_score(0.0, 0.0);
    int epoch = 0, best_epoch = 0;
    double patience = 0;
    string best_file = "";
    REAL_t best_score = 0.0;

    shared_ptr<Visualizer> visualizer;
    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer, FLAGS_visualizer_hostname, FLAGS_visualizer_port);

    auto pred_fun = [&model](vector<uint>& example) {
        graph::NoBackprop nb;
        auto final_states = model.get_final_activation(
            &example, 0.0
        );
        // no softmax needed, simply get best guess
        return model.decode(
            model.embedding[example.back()],
            final_states
        ).argmax();
    };

    // if no training should occur then use the validation set
    // to see how good the loaded model is.
    if (epochs == 0) {
        best_validation_score = SST::average_recall(validation_set, pred_fun, FLAGS_j);
        std::cout << "   Root recall = " << std::get<1>(best_validation_score) << std::endl;
        std::cout << "Overall recall = " << std::get<0>(best_validation_score) << std::endl;
    }

    while (patience < FLAGS_patience && epoch < epochs) {

        if (memory_penalty_curve_type == 1) { // linear
            model.memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
            );
            for (auto& thread_model : thread_models) {
                thread_model.memory_penalty = std::min(
                    FLAGS_memory_penalty,
                    (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
                );
            }
        } else if (memory_penalty_curve_type == 2) { // square
            model.memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
            );
            for (auto& thread_model : thread_models) {
                thread_model.memory_penalty = std::min(
                    FLAGS_memory_penalty,
                    (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
                );
            }
        }

        atomic<int> batches_processed(0);
        ReportProgress<double> journalist(
            utils::MS() << "Epoch " << ++epoch,      // what to say first
            dataset.size() // how many steps to expect before being done with epoch
        );

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool->run([&word_vocab, &visualizer, &thread_embedding_params, &thread_params, &thread_models, batch_id, &journalist, &solver, &embedding_solver, &dataset, &best_validation_score, &batches_processed]() {
                auto& thread_model     = thread_models[ThreadPool::get_thread_number()];
                auto& params           = thread_params[ThreadPool::get_thread_number()];
                auto& embedding_params = thread_embedding_params[ThreadPool::get_thread_number()];
                auto& minibatch        = dataset[batch_id];
                // many forward steps here:
                for (auto & example : minibatch) {
                    auto final_states = thread_model.get_final_activation(&std::get<0>(example), FLAGS_dropout);
                    auto logprobs = thread_model.decode(
                        thread_model.embedding[std::get<0>(example).back()],
                        final_states,
                        FLAGS_dropout
                    );
                    auto error = MatOps<REAL_t>::softmax_cross_entropy_rowwise(logprobs, std::get<1>(example));
                    if (FLAGS_average_gradient)
                        error = error/ minibatch.size();
                    if (std::get<2>(example) && FLAGS_root_weight != 1.0)
                        error = error * FLAGS_root_weight;
                    error.grad();
                    graph::backward(); // backpropagate
                }
                // One step of gradient descent
                solver->step(params);
                // no L2 penalty on embedding:
                embedding_solver->step(embedding_params);

                // show sentiment detection as it happens:
                if (visualizer != nullptr) {
                    // show sentiment detection as system learns:
                    visualizer->throttled_feed(seconds(10), [&word_vocab, &visualizer, &minibatch, &thread_model]() {
                            // pick example
                            auto& example = std::get<0>(minibatch[utils::randint(0, minibatch.size()-1)]);
                            // visualization does not backpropagate.
                            graph::NoBackprop nb;
                            auto final_states = thread_model.get_final_activation(&example, 0.0);
                            // make prediction
                            auto probs = MatOps<REAL_t>::softmax_rowwise(
                                thread_model.decode(
                                    thread_model.embedding[example.back()],
                                    final_states,
                                    0.0
                                )
                            );
                            return json_classification(
                                word_vocab.decode(&example),
                                probs,
                                SST::label_names
                            );
                        }
                    );
                }
                // report minibatch completion to progress bar
                journalist.tick(++batches_processed, std::get<0>(best_validation_score));
            });
        }
        pool->wait_until_idle();
        journalist.done();
        auto new_validation = SST::average_recall(validation_set, pred_fun, FLAGS_j);
        std::cout << "Root recall=" << std::get<1>(new_validation) << std::endl;
        if (solver->method == Solver::METHOD_ADAGRAD) {
            solver->reset_caches(params);
            embedding_solver->reset_caches(embedding_params);
        }
        if (
            (FLAGS_validation_metric == 0 && std::get<0>(new_validation) + 1e-6 < std::get<0>(best_validation_score)) ||
            (FLAGS_validation_metric == 1 && std::get<1>(new_validation) + 1e-6 < std::get<1>(best_validation_score))
            ) {
            // lose patience:
            patience += 1;
        } else {
            // recover some patience:
            patience = std::max(patience - 1, 0.0);
            best_validation_score = new_validation;
        }
        if (best_validation_score != new_validation) {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << std::get<0>(best_validation_score) << "% ("<< std::get<0>(new_validation) << "%), patience = " << patience << std::endl;
        } else {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << std::get<0>(best_validation_score) << "%, patience = " << patience << std::endl;
            best_epoch = epoch;
        }
        if ((FLAGS_validation_metric == 0 && std::get<0>(new_validation) > best_score) ||
            (FLAGS_validation_metric == 1 && std::get<1>(new_validation) > best_score)) {
            best_score = (FLAGS_validation_metric == 0) ?
                std::get<0>(new_validation) :
                std::get<1>(new_validation);
            // save best:
            if (!FLAGS_save_location.empty()) {
                model.save(FLAGS_save_location);
                best_file = FLAGS_save_location;
            }
        }
    }

    if (!FLAGS_test.empty()) {
        auto test_set = SST::convert_trees_to_indexed_minibatches(
            word_vocab,
            SST::load(FLAGS_test),
            FLAGS_minibatch
        );
        if (!FLAGS_save_location.empty() && !best_file.empty()) {
            std::cout << "loading from best validation parameters \"" << best_file << "\"" << std::endl;
            auto params = model.parameters();
            utils::load_matrices(params, best_file);
        }
        auto recall = SST::average_recall(test_set, pred_fun, FLAGS_j);

        std::cout << "Done training" << std::endl;
        std::cout << "Test recall "  << std::get<0>(recall) << "%, root => " << std::get<1>(recall)<< "%" << std::endl;
        if (!FLAGS_results_file.empty()) {
            ofstream fp;
            fp.open(FLAGS_results_file.c_str(), std::ios::out | std::ios::app);
            fp         << FLAGS_solver
               << "\t" << FLAGS_minibatch
               << "\t" << (FLAGS_fast_dropout ? "fast" : "std")
               << "\t" << FLAGS_dropout
               << "\t" << FLAGS_hidden
               << "\t" << std::get<0>(recall)
               << "\t" << std::get<1>(recall)
               << "\t" << best_epoch
               << "\t" << FLAGS_memory_penalty
               << "\t" << FLAGS_memory_penalty_curve;
            if ((FLAGS_solver == "adagrad" || FLAGS_solver == "sgd")) {
                fp << "\t" << FLAGS_learning_rate;
            } else {
                fp << "\t" << "N/A";
            }
            fp  << "\t" << FLAGS_reg << std::endl;
        }
    }
}

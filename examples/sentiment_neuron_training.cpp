#include <iostream>
#include <vector>
#include <fstream>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/tensor/op/spatial.h>
#include <dali/tensor/layers/conv.h>
#include <dali/array/jit/jit.h>
#include <dali/array/expression/computation.h>
#include <dali/array/memory/synchronized_memory.h>
#include <dali/array/memory/memory_bank.h>
#include <dali/array/from_vector.h>
#include <dali/utils/performance_report.h>
#include <dali/utils/concatenate.h>
#include <dali/utils/make_message.h>
#include <dali/utils/timer.h>

#include "third_party/json.hpp"

#include "utils.h"

DEFINE_bool(use_jit_fusion, true, "Whether to use JIT Fusion.");
DEFINE_int32(batch_size, 128, "Batch size");
DEFINE_int32(hidden_size, 256, "Hidden size");
DEFINE_int32(timesteps, 16, "Timesteps");
DEFINE_int32(epochs, 10, "Epochs");
DEFINE_int32(max_fusion_arguments, 3, "Max fusion arguments");
DEFINE_int32(max_examples, 2048, "Max examples");
DEFINE_string(path, utils::dir_join({STR(DALI_EXAMPLES_DATA_DIR), "amazon_reviews", "reviews_Movies_and_TV_5.json"}), "Path to training data");

Tensor uniform_tensor(const shape_t& sizes, DType dtype) {
    int input_size = sizes[0].value();
    return Tensor::uniform(Array(-1.0 / sqrt(input_size), dtype), Array(1.0 / sqrt(input_size), dtype), sizes);
}

Tensor l2_normalize(const Tensor& t, int axis) {
    return t / tensor_ops::L2_norm(t, {axis}, /*keepdims=*/true);
}


struct MLSTM : public AbstractLayer {
    DType dtype_;
    Dim hidden_size_;
    Tensor wx;
    Tensor wh;
    Tensor wmx;
    Tensor wmh;
    Tensor b;
    Tensor gx;
    Tensor gh;
    Tensor gmx;
    Tensor gmh;
    MLSTM(int input_size, int hidden_size, DType dtype) 
        : dtype_(dtype), hidden_size_(hidden_size),
          wx(uniform_tensor({input_size, hidden_size * 4}, dtype)),
          wh(uniform_tensor({hidden_size, hidden_size * 4}, dtype)),
          wmx(uniform_tensor({input_size, hidden_size}, dtype)),
          wmh(uniform_tensor({hidden_size, hidden_size}, dtype)),
          b(uniform_tensor({1, hidden_size * 4}, dtype)),
          gx(uniform_tensor({1, hidden_size * 4}, dtype)),
          gh(uniform_tensor({1, hidden_size * 4}, dtype)),
          gmx(uniform_tensor({1, hidden_size}, dtype)),
          gmh(uniform_tensor({1, hidden_size}, dtype)) {}

    std::tuple<Tensor, Tensor> activate(const Tensor& inputs) const {
        Tensor wx_norm = l2_normalize(wx, 0) * gx;
        Tensor wh_norm = l2_normalize(wh, 0) * gh;
        Tensor wmx_norm = l2_normalize(wmx, 0) * gmx;
        Tensor wmh_norm = l2_normalize(wmh, 0) * gmh;

        auto prev_h = Tensor::zeros({inputs.shape()[0], hidden_size_}, dtype_);
        auto prev_c = Tensor::zeros({inputs.shape()[0], hidden_size_}, dtype_);

        std::vector<Tensor> hs;
        std::vector<Tensor> cs;
        for (int iter = 0; iter < inputs.shape()[1].value(); ++iter) {
            Tensor x = inputs[Slice()][iter];
            auto m = tensor_ops::matmul(x, wmx) * tensor_ops::matmul(prev_h, wmh);
            auto z = tensor_ops::matmul(x, wx) + tensor_ops::matmul(m, wh) + b;
            auto i = tensor_ops::sigmoid(z[Slice()][Slice(0, hidden_size_)]);
            auto f = tensor_ops::sigmoid(z[Slice()][Slice(hidden_size_, 2 * hidden_size_)]);
            auto o = tensor_ops::sigmoid(z[Slice()][Slice(2 * hidden_size_, 3 * hidden_size_)]);
            auto u = tensor_ops::tanh(z[Slice()][Slice(3 * hidden_size_, 4 * hidden_size_)]);
            auto c = f * prev_c + i * u;
            auto h = o * tensor_ops::tanh(c);
            hs.emplace_back(h);
            cs.emplace_back(c);
            prev_h = h;
            prev_c = c;
        }
        return std::make_tuple(tensor_ops::stack(hs, 1), tensor_ops::stack(cs, 1));
    }

    virtual std::vector<Tensor> parameters() const {
        return {wx, wh, wmx, wmh, b, gx, gh, gmx, gmh};
    }
};

struct SentimentNeuronConfig {
    int vocab_size = 256;
    int embedding_size = 64;
    int hidden_size = 256;
};


struct SentimentNeuronModel : public AbstractLayer {
    DType dtype_;
    Layer prediction_head_;
    MLSTM mlstm_;
    Tensor embeddings_;
    SentimentNeuronModel(const SentimentNeuronConfig& config, DType dtype) :
        embeddings_({config.vocab_size, config.embedding_size}, dtype),
        prediction_head_(config.hidden_size, config.vocab_size, dtype),
        mlstm_(config.embedding_size, config.hidden_size, dtype),
        dtype_(dtype) {}
    Tensor activate(const Tensor& x) const {
        auto words = embeddings_[x];
        Tensor hs, cs;
        std::tie(hs, cs) = mlstm_.activate(words);
        auto logits = prediction_head_.activate(hs);
        return logits;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({{embeddings_}, mlstm_.parameters(), prediction_head_.parameters()});
    }
};

std::tuple<Array, Array> training_epoch(const SentimentNeuronModel& model,
                                        std::shared_ptr<solver::AbstractSolver> solver,
                                        Array data,
                                        int batch_size) {
    int num_examples = data.shape()[0].value();
    int timesteps_plus_1 = data.shape()[1].value();
    auto params = model.parameters();
    Array num_correct(0, DTYPE_DOUBLE);
    Array epoch_error(0, DTYPE_DOUBLE);

    for (int batch_start = 0; batch_start < num_examples; batch_start += batch_size) {
        {
            utils::Timer backward("graph");
            auto batch_slice = Slice(batch_start, std::min(batch_start + batch_size, num_examples));
            auto time_slice = Slice(0, timesteps_plus_1 - 1);
            auto time_slice_pred = Slice(1, timesteps_plus_1);
            Tensor x = Array(data[batch_slice][time_slice]);
            Tensor y = Array(data[batch_slice][time_slice_pred]);
            auto probs = model.activate(x);
            Tensor error = tensor_ops::softmax_cross_entropy(probs, y);
            (error.mean()).grad();
            auto batch_error = error.w.sum();
            epoch_error += batch_error;
            {
                utils::Timer backward("backward");
                graph::backward();
            }
            auto op = op::control_dependency(solver->step(params), epoch_error);
            backward.stop();
            op.eval();
        }
    }
    return std::make_tuple(epoch_error / (double)num_examples, num_correct / (double)num_examples);
}


Array load_training_data(const std::string& path, int timesteps, int max_examples) {
    int timesteps_plus_1 = timesteps + 1;
    WithDevicePreference pref(Device::cpu());
    std::ifstream file(path);
    if (file.is_open()) {
        std::string line;
        Array out = Array::zeros({max_examples, timesteps_plus_1}, DTYPE_INT32);
        int examples = 0;
        std::vector<int> sentence;
        while (std::getline(file, line)) {
            auto parsed = nlohmann::json::parse(line);
            for (auto c : parsed.at("reviewText").get<std::string>()) {
                sentence.emplace_back(std::min(int(c), 255));
                if (sentence.size() == timesteps_plus_1) {
                    out[examples] = Array::from_vector(sentence);
                    sentence.clear();
                    out.eval();
                    ++examples;
                    if (examples == max_examples) {
                        break;
                    }
                }
            }
            if (examples == max_examples) {
                break;
            }
        }
        file.close();
        return out;
    } else {
        // could not load file.
        std::cout << "Failed to open \"" << path << "\" generating dummy data instead." << std::endl;
        return op::uniform(0, 255, {max_examples, timesteps_plus_1});
    }
}

int main (int argc, char *argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage("Sentiment Neuron training\n");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_device == -1) {
        default_preferred_device = Device::cpu();
    }
    if (FLAGS_device >= 0) {
        ASSERT2(FLAGS_device < Device::num_gpus(), utils::make_message("Cannot run on GPU ", FLAGS_device, ", only found ", Device::num_gpus(), " gpus."));
        default_preferred_device = Device::gpu(FLAGS_device);
    }
    std::cout << "Running on " << default_preferred_device.description() << "." << std::endl;
    utils::random::set_seed(123123);
    const int batch_size = FLAGS_batch_size;
    op::jit::WithJITFusionPreference jit_pref(FLAGS_use_jit_fusion);
    op::jit::WithJITMaxArguments jit_max_args(FLAGS_max_fusion_arguments);
    std::cout << "Use JIT Fusion = " << (op::jit::jit_fusion_preference() ? "True" : "False") << "." << std::endl;
    std::cout << "DONE." << std::endl;

    SentimentNeuronConfig config;
    config.hidden_size = FLAGS_hidden_size;

    SentimentNeuronModel model(config, DTYPE_FLOAT);
    auto params = model.parameters();
    auto solver = solver::construct("sgd", params, 0.01);
    solver->clip_norm_ = 0.0;
    solver->clip_abs_  = 0.0;
    long prev_number_of_computations = number_of_computations();
    long prev_number_of_allocations = memory::number_of_allocations();
    long prev_number_of_bytes_allocated = memory::number_of_bytes_allocated();
    long prev_actual_number_of_allocations = memory::bank::number_of_allocations();
    long prev_actual_number_of_bytes_allocated = memory::bank::number_of_bytes_allocated();

    {
        std::vector<Array> inits;
        for (auto& p : params) {
            inits.emplace_back(p.w);
        }
        op::control_dependencies(Array(0), inits).eval();
    }

    Array x = load_training_data(FLAGS_path, FLAGS_timesteps, FLAGS_max_examples);

    // for (int i = 1; i < 22; ++i) {
    //     int tstep = i * 16;
        int tstep = FLAGS_timesteps;
        std::cout << "Timesteps " << tstep << std::endl;
        for (int i = 0; i < FLAGS_epochs; i++) {
            auto epoch_start_time = std::chrono::system_clock::now();
            training_epoch(model, solver, x, batch_size);
            std::chrono::duration<double> epoch_duration = (std::chrono::system_clock::now() - epoch_start_time);
            std::cout << epoch_duration.count()
                  << " " << number_of_computations() - prev_number_of_computations
                  << " " << memory::number_of_allocations() - prev_number_of_allocations
                  << " " << memory::number_of_bytes_allocated() - prev_number_of_bytes_allocated
                  << " " << memory::bank::number_of_allocations() - prev_actual_number_of_allocations
                  << " " << memory::bank::number_of_bytes_allocated() - prev_actual_number_of_bytes_allocated
                  << std::endl;
            prev_number_of_computations = number_of_computations();
            prev_number_of_allocations = memory::number_of_allocations();
            prev_number_of_bytes_allocated = memory::number_of_bytes_allocated();
            prev_actual_number_of_allocations = memory::bank::number_of_allocations();
            prev_actual_number_of_bytes_allocated = memory::bank::number_of_bytes_allocated();
        }
        optimization_report();
        utils::Timer::report();
    // }
}

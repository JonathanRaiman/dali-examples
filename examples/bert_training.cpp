#include <iostream>
#include <vector>

#include <dali/core.h>
#include <dali/utils.h>
#include <dali/tensor/op/spatial.h>
#include <dali/tensor/layers/conv.h>
#include <dali/array/jit/jit.h>
#include <dali/array/expression/computation.h>
#include <dali/array/memory/synchronized_memory.h>
#include <dali/array/memory/memory_bank.h>
#include <dali/utils/performance_report.h>
#include <dali/utils/concatenate.h>
#include <dali/utils/make_message.h>
#include <dali/utils/timer.h>

#include "utils.h"

DEFINE_bool(use_jit_fusion, true, "Whether to use JIT Fusion.");
DEFINE_int32(batch_size, 256, "Batch size");
DEFINE_int32(epochs, 2, "Epochs");

// BERT implementation adapted from https://github.com/huggingface/pytorch-pretrained-BERT's PyTorch implementation

std::unordered_map<std::string, std::function<Tensor(Tensor)>> ACT2FN = {
    {"gelu", tensor_ops::gelu},
    {"relu", [](Tensor x) {return tensor_ops::relu(x);}},
    {"swish", tensor_ops::swish},
};

struct BertConfig {
    int hidden_size;
    int vocab_size;
    int intermediate_size;
    int max_position_embeddings;
    int type_vocab_size;
    int num_attention_heads;
    double hidden_dropout_prob;
    double attention_probs_dropout_prob;
    std::string hidden_act;
};

struct BertLayerNorm : public AbstractLayer {
    Tensor gamma_, beta_;
    double variance_epsilon_;
    BertLayerNorm(const BertConfig& config, double variance_epsilon, DType dtype)
        : gamma_(Tensor::ones({config.hidden_size}, dtype)),
          beta_(Tensor::zeros({config.hidden_size}, dtype)),
          variance_epsilon_(variance_epsilon) {}

    Tensor activate(const Tensor& x) const {
        auto u = x.mean({-1}, /*keepdims=*/true);
        auto x_zeroed = (x - u);
        auto s = x_zeroed.square().mean({-1}, /*keepdims=*/true);
        auto x_norm = x_zeroed / (s + variance_epsilon_).sqrt();
        return gamma_ * x + beta_;
    }

    virtual std::vector<Tensor> parameters() const {return {gamma_, beta_};}
};

struct BertEmbeddings : public AbstractLayer {
    Tensor word_embeddings_,
           position_embeddings_,
           token_type_embeddings_;
    BertLayerNorm layer_norm_;
    double hidden_dropout_prob_;
    BertEmbeddings(const BertConfig& config, DType dtype)
        : word_embeddings_(Tensor::uniform(Array(-0.01, dtype), Array(0.01, dtype), {config.vocab_size, config.hidden_size})),
          position_embeddings_(Tensor::uniform(Array(-0.01, dtype), Array(0.01, dtype), {config.vocab_size, config.hidden_size})),
          token_type_embeddings_(Tensor::uniform(Array(-0.01, dtype), Array(0.01, dtype), {config.vocab_size, config.hidden_size})),
          layer_norm_(config, 1e-12, dtype),
          hidden_dropout_prob_(config.hidden_dropout_prob) {}

    Tensor activate(const Tensor& input_ids) const {
        return activate(input_ids, Tensor::zeros({1, 1}));
    }
    
    Tensor activate(const Tensor& input_ids, Tensor token_type_ids) const {
        // input_ids has shape Batch x Time
        auto seq_length = input_ids.shape()[1];
        auto position_ids = op::arange(seq_length).expand_dims(0);

        auto words_embeddings = word_embeddings_[input_ids];
        auto position_embeddings = position_embeddings_[position_ids];
        auto token_type_embeddings = token_type_embeddings_[token_type_ids];
        auto embeddings = words_embeddings + position_embeddings + token_type_embeddings;
        embeddings = layer_norm_.activate(embeddings);
        embeddings = tensor_ops::dropout(embeddings, hidden_dropout_prob_);
        return embeddings;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({layer_norm_.parameters(),
                                   {word_embeddings_, position_embeddings_, token_type_embeddings_}});
    }
};


struct BertSelfAttention : public AbstractLayer {
    int num_attention_heads_, attention_head_size_, all_head_size_;
    Layer query_, key_, value_;
    double attention_probs_dropout_prob_;
    BertSelfAttention(const BertConfig& config, DType dtype)
        : num_attention_heads_(config.num_attention_heads),
          attention_head_size_(config.hidden_size / config.num_attention_heads),
          all_head_size_(config.num_attention_heads * (config.hidden_size / config.num_attention_heads)),
          query_(config.hidden_size, all_head_size_, dtype),
          key_(config.hidden_size, all_head_size_, dtype),
          value_(config.hidden_size, all_head_size_, dtype),
          attention_probs_dropout_prob_(config.attention_probs_dropout_prob) {}

    Tensor transpose_for_scores(const Tensor& x) const {
        shape_t new_x_shape(x.shape().begin(), x.shape().end() - 1);
        new_x_shape.emplace_back(num_attention_heads_);
        new_x_shape.emplace_back(attention_head_size_);
        return x.reshape(new_x_shape).dimshuffle({0, 2, 1, 3});
    }

    Tensor activate(const Tensor& hidden_states, const Tensor& attention_mask) const {
        auto mixed_query_layer = query_.activate(hidden_states);
        auto mixed_key_layer = key_.activate(hidden_states);
        auto mixed_value_layer = value_.activate(hidden_states);
        auto query_layer = transpose_for_scores(mixed_query_layer);
        auto key_layer = transpose_for_scores(mixed_key_layer);
        auto value_layer = transpose_for_scores(mixed_value_layer);

        // Take the dot product between "query" and "key" to get the raw attention scores.
        auto attention_scores = tensor_ops::dot(query_layer, key_layer.swapaxes(-1, -2));
        attention_scores = attention_scores / std::sqrt(attention_head_size_);
        // Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask;

        // Normalize the attention scores to probabilities.
        auto attention_probs = tensor_ops::softmax(attention_scores, /*axis=*/-1);

        // This is actually dropping out entire tokens to attend to, which might
        // seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = tensor_ops::dropout(attention_probs, attention_probs_dropout_prob_);
        auto context_layer = tensor_ops::dot(attention_probs, value_layer);
        context_layer = context_layer.dimshuffle({0, 2, 1, 3});
        shape_t new_context_layer_shape(context_layer.shape().begin(), context_layer.shape().begin() - 2);
        new_context_layer_shape.emplace_back(all_head_size_);
        context_layer = context_layer.reshape(new_context_layer_shape);
        return context_layer;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({query_.parameters(), key_.parameters(), value_.parameters()});
    }
};


struct BertSelfOutput : public AbstractLayer {
    Layer dense_;
    BertLayerNorm layer_norm_;
    double hidden_dropout_prob_;

    BertSelfOutput(const BertConfig& config, DType dtype) :
        dense_(config.hidden_size, config.hidden_size, dtype),
        layer_norm_(config, 1e-12, dtype),
        hidden_dropout_prob_(config.hidden_dropout_prob) {}

    Tensor activate(const Tensor& hidden_states, const Tensor& input_tensor) const {
        auto out = dense_.activate(hidden_states);
        out = tensor_ops::dropout(out, hidden_dropout_prob_);
        out = layer_norm_.activate(out + input_tensor);
        return out;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({dense_.parameters(), layer_norm_.parameters()});
    }
};


struct BertIntermediate : public AbstractLayer {
    Layer dense_;
    std::function<Tensor(Tensor)> intermediate_act_fn_;
    BertIntermediate(const BertConfig& config, DType dtype) :
        dense_(config.hidden_size, config.intermediate_size, dtype),
        intermediate_act_fn_(ACT2FN[config.hidden_act]) {}

    Tensor activate(const Tensor& hidden_states) const {
        return intermediate_act_fn_(dense_.activate(hidden_states));
    }
};

// class BertOutput(nn.Module):
//     def __init__(self, config):
//         super(BertOutput, self).__init__()
//         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
//         self.LayerNorm = BertLayerNorm(config)
//         self.dropout = nn.Dropout(config.hidden_dropout_prob)

//     def forward(self, hidden_states, input_tensor):
//         hidden_states = self.dense(hidden_states)
//         hidden_states = self.dropout(hidden_states)
//         hidden_states = self.LayerNorm(hidden_states + input_tensor)
//         return hidden_states


// class BertLayer(nn.Module):
//     def __init__(self, config):
//         super(BertLayer, self).__init__()
//         self.attention = BertAttention(config)
//         self.intermediate = BertIntermediate(config)
//         self.output = BertOutput(config)

//     def forward(self, hidden_states, attention_mask):
//         attention_output = self.attention(hidden_states, attention_mask)
//         intermediate_output = self.intermediate(attention_output)
//         layer_output = self.output(intermediate_output, attention_output)
//         return layer_output


// class BertEncoder(nn.Module):
//     def __init__(self, config):
//         super(BertEncoder, self).__init__()
//         layer = BertLayer(config)
//         self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

//     def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
//         all_encoder_layers = []
//         for layer_module in self.layer:
//             hidden_states = layer_module(hidden_states, attention_mask)
//             if output_all_encoded_layers:
//                 all_encoder_layers.append(hidden_states)
//         if not output_all_encoded_layers:
//             all_encoder_layers.append(hidden_states)
//         return all_encoder_layers


// class BertPooler(nn.Module):
//     def __init__(self, config):
//         super(BertPooler, self).__init__()
//         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
//         self.activation = nn.Tanh()

//     def forward(self, hidden_states):
//         # We "pool" the model by simply taking the hidden state corresponding
//         # to the first token.
//         first_token_tensor = hidden_states[:, 0]
//         pooled_output = self.dense(first_token_tensor)
//         pooled_output = self.activation(pooled_output)
//         return pooled_output


// class BertPredictionHeadTransform(nn.Module):
//     def __init__(self, config):
//         super(BertPredictionHeadTransform, self).__init__()
//         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
//         self.transform_act_fn = ACT2FN[config.hidden_act] \
//             if isinstance(config.hidden_act, str) else config.hidden_act
//         self.LayerNorm = BertLayerNorm(config)

//     def forward(self, hidden_states):
//         hidden_states = self.dense(hidden_states)
//         hidden_states = self.transform_act_fn(hidden_states)
//         hidden_states = self.LayerNorm(hidden_states)
//         return hidden_states


// class BertLMPredictionHead(nn.Module):
//     def __init__(self, config, bert_model_embedding_weights):
//         super(BertLMPredictionHead, self).__init__()
//         self.transform = BertPredictionHeadTransform(config)

//         # The output weights are the same as the input embeddings, but there is
//         # an output-only bias for each token.
//         self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
//                                  bert_model_embedding_weights.size(0),
//                                  bias=False)
//         self.decoder.weight = bert_model_embedding_weights
//         self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

//     def forward(self, hidden_states):
//         hidden_states = self.transform(hidden_states)
//         hidden_states = self.decoder(hidden_states) + self.bias
//         return hidden_states


// class BertOnlyMLMHead(nn.Module):
//     def __init__(self, config, bert_model_embedding_weights):
//         super(BertOnlyMLMHead, self).__init__()
//         self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

//     def forward(self, sequence_output):
//         prediction_scores = self.predictions(sequence_output)
//         return prediction_scores


// class BertOnlyNSPHead(nn.Module):
//     def __init__(self, config):
//         super(BertOnlyNSPHead, self).__init__()
//         self.seq_relationship = nn.Linear(config.hidden_size, 2)

//     def forward(self, pooled_output):
//         seq_relationship_score = self.seq_relationship(pooled_output)
//         return seq_relationship_score


// class BertPreTrainingHeads(nn.Module):
//     def __init__(self, config, bert_model_embedding_weights):
//         super(BertPreTrainingHeads, self).__init__()
//         self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
//         self.seq_relationship = nn.Linear(config.hidden_size, 2)

//     def forward(self, sequence_output, pooled_output):
//         prediction_scores = self.predictions(sequence_output)
//         seq_relationship_score = self.seq_relationship(pooled_output)
//         return prediction_scores, seq_relationship_score


struct MnistCnn {
    ConvLayer conv1;
    ConvLayer conv2;
    Layer     fc1;
    Layer     fc2;
    bool use_nchw_;

    MnistCnn(bool use_nchw) : use_nchw_(use_nchw) {
        conv1 = ConvLayer(32, 1, 5, 5, 1, 1, PADDING_T_SAME, use_nchw ? "NCHW" : "NHWC");
        conv2 = ConvLayer(64, 32, 5, 5, 1, 1, PADDING_T_SAME, use_nchw ? "NCHW" : "NHWC");
        fc1 = Layer(7 * 7 * 64, 1024);
        fc2 = Layer(1024, 10);
    }

    Tensor activate(Tensor images, float keep_prob) const {
        images = images.reshape(use_nchw_ ? shape_t{-1, 1, 28, 28} : shape_t{-1, 28, 28, 1});
        // shape (B, 1, 28, 28)
        Tensor out = conv1.activate(images).relu();
        out = tensor_ops::max_pool(out, 2, 2, -1, -1, PADDING_T_VALID, use_nchw_ ? "NCHW" : "NHWC");
        // shape (B, 32, 14, 14)
        out = conv2.activate(out).relu();
        out = tensor_ops::max_pool(out, 2, 2, -1, -1, PADDING_T_VALID, use_nchw_ ? "NCHW" : "NHWC");
        // shape (B, 64, 7, 7)
        out = out.reshape({out.shape()[0], 7 * 7 * 64});
        out = fc1.activate(out).relu();
        // shape (B, 1024)
        out = tensor_ops::dropout(out, 1.0 - keep_prob);
        out = fc2.activate(out);
        // shape (B, 10)
        return out;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({conv1.parameters(),
                                   conv2.parameters(),
                                   fc1.parameters(),
                                   fc2.parameters()});
    }
};

std::tuple<Array, Array> training_epoch(const MnistCnn& model,
                                        std::shared_ptr<solver::AbstractSolver> solver,
                                        Tensor images,
                                        Tensor labels,
                                        int batch_size) {
    int num_images = images.shape()[0].value();
    auto params = model.parameters();
    Array num_correct(0, DTYPE_DOUBLE);
    Array epoch_error(0, DTYPE_DOUBLE);
    for (int batch_start = 0; batch_start < num_images; batch_start += batch_size) {
        auto batch_slice = Slice(batch_start, std::min(batch_start + batch_size, num_images));
        Tensor batch_images = images[batch_slice];
        Tensor batch_labels = labels[batch_slice];
        batch_images.constant = true;
        Tensor probs = model.activate(batch_images, 0.5);
        Tensor error = tensor_ops::softmax_cross_entropy(probs, batch_labels);
        error.mean().grad();
        if (!FLAGS_use_jit_fusion) error.w.eval();
        epoch_error += error.w.sum();
        graph::backward();
        op::control_dependencies(solver->step(params), {num_correct, epoch_error}).eval();
    }
    return std::make_tuple(epoch_error / (double)num_images, num_correct / (double)num_images);
}

int main (int argc, char *argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage("BERT training\n");
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
    std::cout << "Use JIT Fusion = " << (op::jit::jit_fusion_preference() ? "True" : "False") << "." << std::endl;
    std::cout << "DONE." << std::endl;
    // BidirectionalTransformer model;
    // auto params = model.parameters();
    // auto solver = solver::construct("sgd", params, 0.01);
    // solver->clip_norm_ = 0.0;
    // solver->clip_abs_  = 0.0;

    // PerformanceReport report;
    // Array epoch_error, epoch_correct;

    // long prev_number_of_computations = number_of_computations();
    // long prev_number_of_allocations = memory::number_of_allocations();
    // long prev_number_of_bytes_allocated = memory::number_of_bytes_allocated();
    // long prev_actual_number_of_allocations = memory::bank::number_of_allocations();
    // long prev_actual_number_of_bytes_allocated = memory::bank::number_of_bytes_allocated();

    // for (int i = 0; i < FLAGS_epochs; ++i) {
    //     auto epoch_start_time = std::chrono::system_clock::now();
    //     std::tie(epoch_error, epoch_correct) = training_epoch(model, solver, train_x, train_y, batch_size);
    //     std::chrono::duration<double> epoch_duration
    //             = (std::chrono::system_clock::now() - epoch_start_time);
    //     default_preferred_device.wait();
    //     std::cout << epoch_duration.count()
    //               << " " << number_of_computations() - prev_number_of_computations
    //               << " " << memory::number_of_allocations() - prev_number_of_allocations
    //               << " " << memory::number_of_bytes_allocated() - prev_number_of_bytes_allocated
    //               << " " << memory::bank::number_of_allocations() - prev_actual_number_of_allocations
    //               << " " << memory::bank::number_of_bytes_allocated() - prev_actual_number_of_bytes_allocated
    //               << std::endl;
    //     prev_number_of_computations = number_of_computations();
    //     prev_number_of_allocations = memory::number_of_allocations();
    //     prev_number_of_bytes_allocated = memory::number_of_bytes_allocated();
    //     prev_actual_number_of_allocations = memory::bank::number_of_allocations();
    //     prev_actual_number_of_bytes_allocated = memory::bank::number_of_bytes_allocated();
    // }
}

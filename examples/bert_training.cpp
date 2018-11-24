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
    int hidden_size = 768;
    int vocab_size = 10;
    int intermediate_size = 3072;
    int max_position_embeddings = 512;
    int type_vocab_size = 2;
    int num_attention_heads = 12;
    int num_hidden_layers = 12;
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;
    double initializer_range = 0.02;
    std::string hidden_act = "gelu";
};

struct BertLayerNorm : public AbstractLayer {
    Tensor gamma_, beta_;
    double variance_epsilon_;
    BertLayerNorm(const BertConfig& config, double variance_epsilon, DType dtype)
        : gamma_(Tensor::normal(Array(0.0, dtype), Array(config.initializer_range, dtype), {config.hidden_size})),
          beta_(Tensor::normal(Array(0.0, dtype), Array(config.initializer_range, dtype), {config.hidden_size})),
          variance_epsilon_(variance_epsilon) {}

    Tensor activate(const Tensor& x) const {
        auto u = x.mean({-1}, /*keepdims=*/true);
        auto x_zeroed = (x - u);
        auto s = x_zeroed.square().mean({-1}, /*keepdims=*/true);
        auto x_norm = x_zeroed / (s + variance_epsilon_).sqrt();
        return gamma_[NewAxis()][NewAxis()] * x + beta_[NewAxis()][NewAxis()];
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
        : word_embeddings_(Tensor::normal(Array(0.0, dtype), Array(config.initializer_range, dtype), {config.vocab_size, config.hidden_size})),
          position_embeddings_(Tensor::normal(Array(0.0, dtype), Array(config.initializer_range, dtype), {config.vocab_size, config.hidden_size})),
          token_type_embeddings_(Tensor::normal(Array(0.0, dtype), Array(config.initializer_range, dtype), {config.vocab_size, config.hidden_size})),
          layer_norm_(config, 1e-12, dtype),
          hidden_dropout_prob_(config.hidden_dropout_prob) {}

    Tensor activate(const Tensor& input_ids) const {
        return activate(input_ids, Tensor::zeros({1, 1}));
    }
    
    Tensor activate(const Tensor& input_ids, Tensor token_type_ids) const {
        ASSERT2(input_ids.ndim() == 2, utils::make_message("Expected input_ids to have ndim = 2, but got input_ids.shape = ", input_ids.shape(), "."));
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
        auto attention_scores = tensor_ops::matmul(query_layer, key_layer.swapaxes(-1, -2));
        // TODO(jonathan): the output of matmul is different than pyTorch's expected broadcasting
        // behavior... investigate
        attention_scores = attention_scores / std::sqrt(attention_head_size_);
        // Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask;

        // Normalize the attention scores to probabilities.
        auto attention_probs = tensor_ops::softmax(attention_scores, /*axis=*/-1);

        // This is actually dropping out entire tokens to attend to, which might
        // seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = tensor_ops::dropout(attention_probs, attention_probs_dropout_prob_);
        auto context_layer = tensor_ops::matmul(attention_probs, value_layer);
        context_layer = context_layer.dimshuffle({0, 2, 1, 3});
        shape_t new_context_layer_shape(context_layer.shape().begin(), context_layer.shape().end() - 2);
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
        return layer_norm_.activate(out + input_tensor);
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({dense_.parameters(), layer_norm_.parameters()});
    }
};

struct BertAttention : public AbstractLayer {
    BertSelfAttention self_;
    BertSelfOutput output_;
    BertAttention(const BertConfig& config, DType dtype) :
        self_(config, dtype), output_(config, dtype) {}
    
    Tensor activate(const Tensor& input_tensor, const Tensor& attention_mask) const {
        auto self_output = self_.activate(input_tensor, attention_mask);
        auto attention_output = output_.activate(self_output, input_tensor);
        return attention_output;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({self_.parameters(), output_.parameters()});
    }
};

struct BertIntermediate : public AbstractLayer {
    Layer dense_;
    std::function<Tensor(Tensor)> intermediate_act_fn_;
    BertIntermediate(const BertConfig& config, DType dtype) :
        dense_(config.hidden_size, config.intermediate_size, dtype),
        intermediate_act_fn_(ACT2FN.at(config.hidden_act)) {}

    Tensor activate(const Tensor& hidden_states) const {
        return intermediate_act_fn_(dense_.activate(hidden_states));
    }

    virtual std::vector<Tensor> parameters() const {return dense_.parameters();}
};

struct BertOutput : public AbstractLayer {
    Layer dense_;
    BertLayerNorm layer_norm_;
    double hidden_dropout_prob_;

    BertOutput(const BertConfig& config, DType dtype) :
        dense_(config.intermediate_size, config.hidden_size, dtype),
        layer_norm_(config, 1e-12, dtype),
        hidden_dropout_prob_(config.hidden_dropout_prob) {}

    Tensor activate(const Tensor& hidden_states, const Tensor& input_tensor) const {
        auto out = dense_.activate(hidden_states);
        out = tensor_ops::dropout(out, hidden_dropout_prob_);
        return layer_norm_.activate(out + input_tensor);
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({dense_.parameters(), layer_norm_.parameters()});
    }
};

struct BertLayer : public AbstractLayer {
    BertAttention attention_;
    BertIntermediate intermediate_;
    BertOutput output_;
    BertLayer(const BertConfig& config, DType dtype) :
        attention_(config, dtype),
        intermediate_(config, dtype),
        output_(config, dtype) {}

    Tensor activate(const Tensor& hidden_states, const Tensor& attention_mask) const {
        auto attention_output = attention_.activate(hidden_states, attention_mask);
        auto intermediate_output = intermediate_.activate(attention_output);
        return output_.activate(intermediate_output, attention_output);
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({attention_.parameters(), intermediate_.parameters(), output_.parameters()});
    }
};

struct BertEncoder : public AbstractLayer {
    std::vector<BertLayer> layers_;
    BertEncoder(const BertConfig& config, DType dtype) {
        for (int i = 0; i < config.num_hidden_layers; i++) {
            layers_.emplace_back(config, dtype);
        }
    }

    std::vector<Tensor> activate(const Tensor& hidden_states, const Tensor& attention_mask, bool output_all_encoded_layers) const {
        std::vector<Tensor> all_encoder_layers;
        Tensor out = hidden_states;
        for (auto& layer : layers_) {
            out = layer.activate(out, attention_mask);
            if (output_all_encoded_layers) {
                all_encoder_layers.emplace_back(out);
            }
        }
        if (!output_all_encoded_layers) {
            all_encoder_layers.emplace_back(out);
        }
        return all_encoder_layers;
    }

    virtual std::vector<Tensor> parameters() const {
        std::vector<Tensor> params;
        for (auto& layer : layers_) {
            auto layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

struct BertPooler : public AbstractLayer {
    Layer dense_;
    BertPooler(const BertConfig& config, DType dtype) : dense_(config.hidden_size, config.hidden_size, dtype) {}
    Tensor activate(const Tensor& hidden_states) const {
        // We "pool" the model by simply taking the hidden state corresponding
        // to the first token.
        return dense_.activate(hidden_states[Slice()][0]).tanh();
    }
    virtual std::vector<Tensor> parameters() const {return dense_.parameters();}
};

struct BertPredictionHeadTransform : public AbstractLayer {
    Layer dense_;
    std::function<Tensor(Tensor)> transform_act_fn_;
    BertLayerNorm layer_norm_;
    BertPredictionHeadTransform(const BertConfig& config, DType dtype) :
        dense_(config.hidden_size, config.hidden_size, dtype),
        transform_act_fn_(ACT2FN.at(config.hidden_act)),
        layer_norm_(config, 1e-12, dtype) {}
    Tensor activate(const Tensor& hidden_states) const {
        return layer_norm_.activate(transform_act_fn_(dense_.activate(hidden_states)));
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({dense_.parameters(), layer_norm_.parameters()});
    }
};

struct BertLMPredictionHead : public AbstractLayer {
    BertPredictionHeadTransform transform_;
    Layer decoder_;
    BertLMPredictionHead(const BertConfig& config, const Tensor& bert_model_embedding_weights, DType dtype)
        : transform_(config, dtype),
          decoder_(bert_model_embedding_weights.shape()[1].value(),
                   bert_model_embedding_weights.shape()[0].value(), dtype) {
        // TODO(jonathan): note that this parameter will be sent twice to the optimizer...
        decoder_.W_ = bert_model_embedding_weights;
    }

    Tensor activate(const Tensor& hidden_states) const {
        return decoder_.activate(transform_.activate(hidden_states));
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({transform_.parameters(), decoder_.parameters()});
    }
};

struct BertOnlyMLMHead : public AbstractLayer {
    BertLMPredictionHead predictions_;
    BertOnlyMLMHead(const BertConfig& config, const Tensor& bert_model_embedding_weights, DType dtype) :
        predictions_(config, bert_model_embedding_weights, dtype) {}
    
    Tensor activate(const Tensor& hidden_states) const {
        return predictions_.activate(hidden_states);
    }

    virtual std::vector<Tensor> parameters() const {return predictions_.parameters();}
};

struct BertOnlyNSPHead : public AbstractLayer {
    Layer seq_relationship_;
    BertOnlyNSPHead(const BertConfig& config, DType dtype) : seq_relationship_(config.hidden_size, 2, dtype) {}
    Tensor activate(const Tensor& pooled_output) const {return seq_relationship_.activate(pooled_output);}
    virtual std::vector<Tensor> parameters() const {return seq_relationship_.parameters();}
};


struct BertPreTrainingHeads : public AbstractLayer {
    BertLMPredictionHead predictions_;
    Layer seq_relationship_;
    BertPreTrainingHeads(const BertConfig& config, const Tensor& bert_model_embedding_weights, DType dtype) :
        predictions_(config, bert_model_embedding_weights, dtype),
        seq_relationship_(config.hidden_size, 2, dtype) {}

    std::tuple<Tensor, Tensor> activate(const Tensor& sequence_output, const Tensor& pooled_output) const {
        auto prediction_scores = predictions_.activate(sequence_output);
        auto seq_relationship_score = seq_relationship_.activate(pooled_output);
        return std::make_tuple(prediction_scores, seq_relationship_score);
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({predictions_.parameters(), seq_relationship_.parameters()});
    }
};

struct BertModel : public AbstractLayer {
    // BERT model ("Bidirectional Embedding Representations from a Transformer").
    BertEmbeddings embeddings_;
    BertEncoder encoder_;
    BertPooler pooler_;
    DType dtype_;
    BertModel(const BertConfig& config, DType dtype) :
        embeddings_(config, dtype),
        encoder_(config, dtype),
        pooler_(config, dtype),
        dtype_(dtype) {}
    std::tuple<std::vector<Tensor>, Tensor> activate(const Tensor& input_ids, const Tensor& token_type_ids,
                                                     const Tensor& attention_mask, bool output_all_encoded_layers) const {
        //     if attention_mask is None:
        //         attention_mask = torch.ones_like(input_ids)
        //     if token_type_ids is None:
        //         token_type_ids = torch.zeros_like(input_ids)
        // We create a 3D attention mask from a 2D tensor mask.
        // Sizes are [batch_size, 1, 1, to_seq_length]
        // So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        // this attention mask is more simple than the triangular masking of causal attention
        // used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        auto extended_attention_mask = attention_mask.expand_dims(1).expand_dims(2);
        // Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        // masked positions, this operation will create a tensor which is 0.0 for
        // positions we want to attend and -10000.0 for masked positions.
        // Since we are adding it to the raw scores before the softmax, this is
        // effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.astype(dtype_);
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0;

        auto embedding_output = embeddings_.activate(input_ids, token_type_ids);
        auto encoded_layers = encoder_.activate(embedding_output, extended_attention_mask, output_all_encoded_layers);
        auto sequence_output = encoded_layers.back();
        auto pooled_output = pooler_.activate(sequence_output);
        return std::make_tuple(encoded_layers, pooled_output);
    }
    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({embeddings_.parameters(), encoder_.parameters(), pooler_.parameters()});
    }
};


std::tuple<Array, Array> training_epoch(const BertModel& model,
                                        std::shared_ptr<solver::AbstractSolver> solver,
                                        Tensor input_ids,
                                        Tensor token_type_ids,
                                        Tensor attention_mask,
                                        int batch_size) {
    int num_examples = input_ids.shape()[0].value();
    auto params = model.parameters();
    Array num_correct(0, DTYPE_DOUBLE);
    Array epoch_error(0, DTYPE_DOUBLE);
    for (int batch_start = 0; batch_start < num_examples; batch_start += batch_size) {
        auto batch_slice = Slice(batch_start, std::min(batch_start + batch_size, num_examples));
        Tensor batch_input_ids = input_ids[batch_slice];
        Tensor batch_token_type_ids = token_type_ids[batch_slice];
        Tensor batch_attention_mask = attention_mask[batch_slice];
        auto probs = model.activate(batch_input_ids, batch_token_type_ids, batch_attention_mask, false);
        // Tensor error = tensor_ops::softmax_cross_entropy(probs, batch_labels);
        // error.mean().grad();
        // if (!FLAGS_use_jit_fusion) error.w.eval();
        // epoch_error += error.w.sum();
        // graph::backward();
        // op::control_dependencies(solver->step(params), {num_correct, epoch_error}).eval();
    }
    return std::make_tuple(epoch_error / (double)num_examples, num_correct / (double)num_examples);
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

    BertConfig config;
    config.num_hidden_layers = 8;
    config.num_attention_heads = 4;
    config.intermediate_size = 1024;
    config.hidden_size = 512;
    config.vocab_size = 32000;
    config.hidden_act = "gelu";

    BertModel model(config, DTYPE_FLOAT);

    auto res = model.activate(Tensor::zeros({100, 10}, DTYPE_INT32),
                              Tensor::zeros({100, 10}, DTYPE_INT32),
                              Tensor::ones({100, 10}, DTYPE_INT32),
                              false);
    std::get<1>(res).print();

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

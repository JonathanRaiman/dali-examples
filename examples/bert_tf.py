from collections import namedtuple
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from os.path import realpath, join, dirname
import time
from argparse_utils import add_bool_flag

BertModel = namedtuple("BertModel", "input_ids token_type_ids attention_mask output labels train_op loss")
SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    return x * tf.nn.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "swish": swish,
    "relu": tf.nn.relu
}


class BertConfig(object):
    def __init__(self,
                 hidden_size=768,
                 vocab_size=10,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 num_attention_heads=12,
                 num_hidden_layers=12,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 hidden_act="gelu"):
        for key, value in locals().items():
            setattr(self, key, value)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = (self.num_attention_heads * (self.hidden_size // self.num_attention_heads))


def bert_layer_norm(config, x, variance_epsilon=1e-12):
    with tf.variable_scope("LayerNorm"):
        gamma_ = tf.get_variable("gamma", shape=[config.hidden_size], dtype=tf.float32)
        beta_ = tf.get_variable("beta", shape=[config.hidden_size], dtype=tf.float32)
        u = tf.reduce_mean(x, axis=-1, keepdims=True)
        x_zeroed = (x - u)
        s = tf.reduce_mean(tf.square(x_zeroed), axis=-1, keepdims=True)
        x_norm = x_zeroed / tf.sqrt(s + variance_epsilon)
        return gamma_[None, None] * x_norm + beta_[None, None]


def bert_embeddings(config, input_ids, token_type_ids):
    word_embeddings_ = tf.get_variable("word_embeddings",
                                       shape=[config.vocab_size, config.hidden_size], dtype=tf.float32)
    position_embeddings_ = tf.get_variable("position_embeddings",
                                           shape=[config.vocab_size, config.hidden_size], dtype=tf.float32)
    token_type_embeddings_ = tf.get_variable("token_type_embeddings",
                                             shape=[config.vocab_size, config.hidden_size], dtype=tf.float32)
    # input_ids has shape Batch x Time
    seq_length = tf.shape(input_ids)[1]
    position_ids = tf.range(seq_length)[None]
    words_embeddings = tf.nn.embedding_lookup(word_embeddings_, input_ids)
    position_embeddings = tf.nn.embedding_lookup(position_embeddings_, position_ids)
    token_type_embeddings = tf.nn.embedding_lookup(token_type_embeddings_, token_type_ids)
    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = bert_layer_norm(config, embeddings)
    embeddings = tf.nn.dropout(embeddings, keep_prob=1.0 - config.hidden_dropout_prob)
    return embeddings


def transpose_for_scores(config, x):
    new_x_shape = ([tf.shape(x)[i] for i in range(len(x.get_shape()) - 1)] +
                   [config.num_attention_heads, config.attention_head_size])
    return tf.transpose(tf.reshape(x, new_x_shape), (0, 2, 1, 3))


def bert_self_attention(config, hidden_states, attention_mask):
    with tf.variable_scope("BertSelfAttention"):
        mixed_query_layer = layers.fully_connected(hidden_states,
                                                   config.hidden_size, scope="FCquery", activation_fn=None)
        mixed_key_layer = layers.fully_connected(hidden_states,
                                                 config.hidden_size, scope="FCkey", activation_fn=None)
        mixed_value_layer = layers.fully_connected(hidden_states,
                                                   config.hidden_size, scope="FCvalue", activation_fn=None)
        query_layer = transpose_for_scores(config, mixed_query_layer)
        key_layer = transpose_for_scores(config, mixed_key_layer)
        value_layer = transpose_for_scores(config, mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, [0, 1, 3, 2]))
        # TODO(jonathan): the output of matmul is different than pyTorch's expected broadcasting
        # behavior... investigate
        attention_scores = attention_scores / np.sqrt(config.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = tf.nn.dropout(attention_probs, keep_prob=1.0 - config.attention_probs_dropout_prob)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = [tf.shape(context_layer)[i] for i in range(2)] + [config.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
    return context_layer


def bert_self_output(config, hidden_states, input_tensor):
    with tf.variable_scope("BertSelfOutput"):
        out = layers.fully_connected(hidden_states, config.hidden_size, scope="dense", activation_fn=None)
        out = tf.nn.dropout(out, keep_prob=1.0 - config.hidden_dropout_prob)
        return bert_layer_norm(config, out + input_tensor)


def bert_attention(config, input_tensor, attention_mask):
    self_output = bert_self_attention(config, input_tensor, attention_mask)
    attention_output = bert_self_output(config, self_output, input_tensor)
    return attention_output


def bert_intermediate(config, hidden_states):
    with tf.variable_scope("BertIntermediate"):
        return layers.fully_connected(hidden_states, config.intermediate_size,
                                      activation_fn=ACT2FN[config.hidden_act], scope="dense")


def bert_output(config, hidden_states, input_tensor):
    with tf.variable_scope("BertOutput"):
        out = layers.fully_connected(hidden_states, config.hidden_size, activation_fn=None)
        out = tf.nn.dropout(out, keep_prob=1.0 - config.hidden_dropout_prob)
        return bert_layer_norm(config, out + input_tensor)


def bert_layer(config, hidden_states, attention_mask):
    attention_output = bert_attention(config, hidden_states, attention_mask)
    intermediate_output = bert_intermediate(config, attention_output)
    return bert_output(config, intermediate_output, attention_output)


def bert_encoder(config, hidden_states, attention_mask, output_all_encoded_layers):
    all_encoder_layers = []
    out = hidden_states
    for i in range(config.num_hidden_layers):
        with tf.variable_scope("Layer{}".format(i)):
            out = bert_layer(config, out, attention_mask)
        if output_all_encoded_layers:
            all_encoder_layers.append(out)
    if not output_all_encoded_layers:
        all_encoder_layers.append(out)
    return all_encoder_layers


def bert_pooler(config, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    with tf.variable_scope("BertPooler"):
        return layers.fully_connected(hidden_states[:, 0], config.hidden_size, scope="Dense", activation_fn=tf.tanh)


# struct BertPredictionHeadTransform : public AbstractLayer {
#     Layer dense_;
#     std::function<Tensor(Tensor)> transform_act_fn_;
#     BertLayerNorm layer_norm_;
#     BertPredictionHeadTransform(const BertConfig& config, DType dtype) :
#         dense_(config.hidden_size, config.hidden_size, dtype),
#         transform_act_fn_(ACT2FN.at(config.hidden_act)),
#         layer_norm_(config, 1e-12, dtype) {}
#     Tensor activate(const Tensor& hidden_states) const {
#         return layer_norm_.activate(transform_act_fn_(dense_.activate(hidden_states)));
#     }

#     virtual std::vector<Tensor> parameters() const {
#         return utils::concatenate({dense_.parameters(), layer_norm_.parameters()});
#     }
# };

# struct BertLMPredictionHead : public AbstractLayer {
#     BertPredictionHeadTransform transform_;
#     Layer decoder_;
#     BertLMPredictionHead(const BertConfig& config, const Tensor& bert_model_embedding_weights, DType dtype)
#         : transform_(config, dtype),
#           decoder_(bert_model_embedding_weights.shape()[1].value(),
#                    bert_model_embedding_weights.shape()[0].value(), dtype) {
#         // TODO(jonathan): note that this parameter will be sent twice to the optimizer...
#         decoder_.W_ = bert_model_embedding_weights;
#     }

#     Tensor activate(const Tensor& hidden_states) const {
#         return decoder_.activate(transform_.activate(hidden_states));
#     }

#     virtual std::vector<Tensor> parameters() const {
#         return utils::concatenate({transform_.parameters(), decoder_.parameters()});
#     }
# };

# struct BertOnlyMLMHead : public AbstractLayer {
#     BertLMPredictionHead predictions_;
#     BertOnlyMLMHead(const BertConfig& config, const Tensor& bert_model_embedding_weights, DType dtype) :
#         predictions_(config, bert_model_embedding_weights, dtype) {}
#     Tensor activate(const Tensor& hidden_states) const {
#         return predictions_.activate(hidden_states);
#     }

#     virtual std::vector<Tensor> parameters() const {return predictions_.parameters();}
# };

# struct BertOnlyNSPHead : public AbstractLayer {
#     Layer seq_relationship_;
#     BertOnlyNSPHead(const BertConfig& config, DType dtype) : seq_relationship_(config.hidden_size, 2, dtype) {}
#     Tensor activate(const Tensor& pooled_output) const {return seq_relationship_.activate(pooled_output);}
#     virtual std::vector<Tensor> parameters() const {return seq_relationship_.parameters();}
# };


# struct BertPreTrainingHeads : public AbstractLayer {
#     BertLMPredictionHead predictions_;
#     Layer seq_relationship_;
#     BertPreTrainingHeads(const BertConfig& config, const Tensor& bert_model_embedding_weights, DType dtype) :
#         predictions_(config, bert_model_embedding_weights, dtype),
#         seq_relationship_(config.hidden_size, 2, dtype) {}

#     std::tuple<Tensor, Tensor> activate(const Tensor& sequence_output, const Tensor& pooled_output) const {
#         auto prediction_scores = predictions_.activate(sequence_output);
#         auto seq_relationship_score = seq_relationship_.activate(pooled_output);
#         return std::make_tuple(prediction_scores, seq_relationship_score);
#     }

#     virtual std::vector<Tensor> parameters() const {
#         return utils::concatenate({predictions_.parameters(), seq_relationship_.parameters()});
#     }
# };


def bert_model(config, input_ids, token_type_ids, attention_mask, output_all_encoded_layers):
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids)
    #     if token_type_ids is None:
    #         token_type_ids = torch.zeros_like(input_ids)
    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    embedding_output = bert_embeddings(config, input_ids, token_type_ids)
    encoded_layers = bert_encoder(config, embedding_output, extended_attention_mask,
                                  output_all_encoded_layers=output_all_encoded_layers)
    sequence_output = encoded_layers[-1]
    pooled_output = bert_pooler(config, sequence_output)
    return (encoded_layers, pooled_output)


def build_model(hidden_layers):
    config = BertConfig(
        num_hidden_layers=hidden_layers,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_size=512,
        vocab_size=32000,
        hidden_act="gelu"
    )
    input_ids = tf.placeholder(tf.int32, [None, None])
    token_type_ids = tf.placeholder(tf.int32, [None, None])
    attention_mask = tf.placeholder(tf.int32, [None, None])
    labels = tf.placeholder(tf.int32, [None, None])
    (last_layer,), output = bert_model(config, input_ids, token_type_ids, attention_mask, False)

    predictions = tf.contrib.layers.fully_connected(last_layer, config.vocab_size, activation_fn=None)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=labels)
    mean_loss = tf.reduce_mean(loss)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(mean_loss)
    loss = tf.reduce_sum(loss)
    return BertModel(input_ids, token_type_ids, attention_mask, output, labels, train_op, loss)


def training_epoch(session, model, input_ids, token_type_ids, attention_mask, batch_size):
    num_examples = len(input_ids)
    num_correct = 0.0
    epoch_error = 0.0
    for batch_start in range(0, num_examples, batch_size):
        batch_slice = slice(batch_start, min(batch_start + batch_size, num_examples))
        batch_input_ids = input_ids[batch_slice]
        batch_token_type_ids = token_type_ids[batch_slice]
        batch_attention_mask = attention_mask[batch_slice]
        _, batch_loss = session.run(
            (model.train_op, model.loss),
            {model.input_ids: batch_input_ids,
             model.labels: batch_input_ids,
             model.token_type_ids: batch_token_type_ids,
             model.attention_mask: batch_attention_mask})
        epoch_error += batch_loss
    return epoch_error / num_examples, num_correct / num_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hidden_layers", default=8, type=int)
    parser.add_argument("--timesteps", default=10, type=int)
    add_bool_flag(parser, "inference_only", True)
    args = parser.parse_args()
    model = build_model(args.hidden_layers)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for iteration in range(args.epochs):
        t0 = time.time()
        if args.inference_only:
            sess.run(model.output,
                     {model.input_ids: np.zeros((args.batch_size, args.timesteps), dtype=np.int32),
                      model.token_type_ids: np.zeros((args.batch_size, args.timesteps), dtype=np.int32),
                      model.attention_mask: np.ones((args.batch_size, args.timesteps), dtype=np.int32)})
        else:
            training_epoch(session=sess,
                           model=model,
                           input_ids=np.zeros((args.batch_size * 10, args.timesteps), dtype=np.int32),
                           token_type_ids=np.zeros((args.batch_size * 10, args.timesteps), dtype=np.int32),
                           attention_mask=np.ones((args.batch_size * 10, args.timesteps), dtype=np.int32),
                           batch_size=args.batch_size)
        t1 = time.time()
        print("%.3f" % (t1 - t0))


if __name__ == "__main__":
    main()

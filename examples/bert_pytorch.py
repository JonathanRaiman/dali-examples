from collections import namedtuple
import argparse
import torch
import torch.nn as nn
import numpy as np
from os.path import realpath, join, dirname
import time
import copy
import math
from argparse_utils import add_bool_flag

from bert_tf import batch_iterator, BertConfig

BertModel = namedtuple("BertModel", "input_ids token_type_ids attention_mask output labels train_op loss")
SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "swish": swish,
    "relu": torch.relu
}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.predictions_layer = nn.Linear(config.hidden_size, config.vocab_size)
        self.cross_ent_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, token_type_ids, attention_mask):
        embedding_output = self.embed(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            torch.unsqueeze(torch.unsqueeze(attention_mask, 1), 2).to(embedding_output.dtype),
            output_all_encoded_layers=False)
        last_layer = encoded_layers[-1]
        output = self.pooler(last_layer) # noqa
        logits = self.predictions_layer(last_layer)
        loss = self.cross_ent_loss(
            torch.reshape(logits,
                          [logits.shape[0] * logits.shape[1], logits.shape[2]]),
            torch.reshape(labels,
                          [labels.shape[0] * labels.shape[1]]))
        return loss


def build_model(hidden_layers):
    config = BertConfig(
        num_hidden_layers=hidden_layers,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_size=512,
        vocab_size=32000,
        hidden_act="gelu"
    )
    return BERT(config)


def training_epoch(model, data_dir, batch_size, max_length, max_examples):
    num_examples = 0
    num_correct = 0.0
    epoch_error = 0.0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for batch in batch_iterator(data_dir, batch_size, max_length):
        batch_input_ids = np.concatenate([np.ones((len(batch), 1)), np.maximum(batch[:, :-1], 0)], axis=-1)
        batch_label_ids = torch.LongTensor(np.maximum(batch, 0))
        batch_attention_mask = torch.LongTensor(batch_input_ids == -1)
        batch_input_ids = torch.LongTensor(batch_input_ids)
        if torch.cuda.is_available():
            batch_input_ids = batch_input_ids.cuda()
            batch_label_ids = batch_label_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()

        batch_loss = model(input_ids=batch_input_ids,
                           labels=batch_label_ids,
                           token_type_ids=batch_input_ids,
                           attention_mask=batch_attention_mask)
        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_error += float(batch_loss)
        num_examples += len(batch)
        if num_examples > max_examples:
            break
    return epoch_error / num_examples, num_correct / num_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hidden_layers", default=8, type=int)
    parser.add_argument("--timesteps", default=10, type=int)
    parser.add_argument("--data_dir", default=join(DATA_DIR, "lm1b"), type=str)
    parser.add_argument("--max_examples", default=2048, type=int)
    add_bool_flag(parser, "inference_only", True)
    args = parser.parse_args()
    model = build_model(args.hidden_layers)
    if torch.cuda.is_available():
        model.cuda()

    for iteration in range(args.epochs):
        t0 = time.time()
        if args.inference_only:
            input_ids = torch.LongTensor(np.zeros((args.batch_size, args.timesteps), dtype=np.int32))
            labels = torch.LongTensor(np.zeros((args.batch_size, args.timesteps), dtype=np.int32))
            token_type_ids = torch.LongTensor(np.zeros((args.batch_size, args.timesteps), dtype=np.int32))
            attention_mask = torch.LongTensor(np.zeros((args.batch_size, args.timesteps), dtype=np.int32))
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()

            with torch.no_grad():
                loss = model(input_ids=input_ids,
                             labels=labels,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask)
                torch.sum(loss)
            # TODO: nogradient
        else:
            training_epoch(model=model,
                           data_dir=args.data_dir,
                           batch_size=args.batch_size,
                           max_length=args.timesteps,
                           max_examples=args.max_examples)
        t1 = time.time()
        print("%.3f" % (t1 - t0))


if __name__ == "__main__":
    main()

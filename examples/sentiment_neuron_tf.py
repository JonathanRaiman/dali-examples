import argparse
import tensorflow as tf
import numpy as np
import json
import time
from os.path import realpath, join, dirname, exists

SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


class HParams(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def embd(X, ndim, nvocab, scope='embedding'):
    with tf.variable_scope(scope):
        embd = tf.get_variable("embedding", [nvocab, ndim])
        h = tf.nn.embedding_lookup(embd, X)
        return h


def fc(x, nout, act, wn=False, bias=True, scope='fc'):
    with tf.variable_scope(scope):
        nin = x.get_shape()[-1].value
        w = tf.get_variable("w", [nin, nout])
        if wn:
            g = tf.get_variable("g", [nout])
        if wn:
            w = tf.nn.l2_normalize(w, axis=0) * g
        z = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("b", [nout])
            z = z + b
        h = act(z)
        return h


def mlstm(inputs, hidden_size, scope='lstm', wn=False):
    timesteps = tf.shape(inputs)[1]
    batch_size = tf.shape(inputs)[0]
    nin = inputs[0].get_shape()[1].value
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, hidden_size * 4])
        wh = tf.get_variable("wh", [hidden_size, hidden_size * 4])
        wmx = tf.get_variable("wmx", [nin, hidden_size])
        wmh = tf.get_variable("wmh", [hidden_size, hidden_size])
        b = tf.get_variable("b", [hidden_size * 4])
        if wn:
            gx = tf.get_variable("gx", [hidden_size * 4])
            gh = tf.get_variable("gh", [hidden_size * 4])
            gmx = tf.get_variable("gmx", [hidden_size])
            gmh = tf.get_variable("gmh", [hidden_size])

    if wn:
        wx = tf.nn.l2_normalize(wx, axis=0) * gx
        wh = tf.nn.l2_normalize(wh, axis=0) * gh
        wmx = tf.nn.l2_normalize(wmx, axis=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, axis=0) * gmh

    def cond(iteration, *args):
        return tf.less(iteration, timesteps)

    def body(iteration, all_hs, all_cs, prev_h, prev_c):
        x = inputs[:, iteration, :]
        m = tf.matmul(x, wmx) * tf.matmul(prev_h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * prev_c + i * u
        h = o * tf.tanh(c)

        new_all_hs = all_hs.write(iteration, h)
        new_all_cs = all_cs.write(iteration, c)
        return iteration + 1, new_all_hs, new_all_cs, h, c

    all_hs_arr = tf.TensorArray(tf.float32, dynamic_size=True, size=0)
    all_cs_arr = tf.TensorArray(tf.float32, dynamic_size=True, size=0)
    init_h = tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)
    init_c = tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)
    init_iteration = tf.zeros([], tf.int32)
    _, all_hs_out, all_cs_out, hfinal, cfinal = tf.while_loop(
        cond, body, loop_vars=[init_iteration, all_hs_arr, all_cs_arr, init_h, init_c])
    hs = tf.transpose(all_hs_out.stack(), (1, 0, 2))
    cs = tf.transpose(all_cs_out.stack(), (1, 0, 2))
    return hs, cs, cfinal, hfinal


def model(hps, X, M=None, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        words = embd(X, hps.nembd, nvocab=hps.nvocab)
        hs, cells, cfinal, hfinal = mlstm(
            words, hidden_size=hps.nhidden, scope='rnn', wn=hps.rnn_wn)
        hs = tf.reshape(hs, [-1, hps.nhidden])
        timesteps = tf.shape(words)[1]
        batch_size = tf.shape(words)[0]
        logits = tf.reshape(fc(
            hs, hps.nvocab, act=lambda x: x, wn=hps.out_wn, scope='out'),
            [batch_size, timesteps, hps.nvocab])
    states = tf.stack([cfinal, hfinal], 0)
    return cells, states, logits


def load_training_data(path, timesteps, max_examples):
    timesteps_plus_1 = timesteps + 1
    if exists(path):
        out = np.zeros((max_examples, timesteps_plus_1), dtype=np.int32)
        sentence = []
        examples = 0
        with open(path, "rt") as fin:
            for line in fin:
                parsed = json.loads(line)
                for c in parsed["reviewText"]:
                    sentence.append(min(ord(c), 255))
                    if len(sentence) == timesteps_plus_1:
                        out[examples] = sentence
                        sentence.clear()
                        examples += 1
                        if examples == max_examples:
                            break
                if examples == max_examples:
                    break
        return out
    else:
        print("Failed to open \"{}\" generating dummy data instead.".format(path))
        return np.random.randint(0, 255, size=(max_examples, timesteps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=2048)
    parser.add_argument("--path", type=str, default=join(DATA_DIR, "amazon_reviews", "reviews_Movies_and_TV_5.json"))
    args = parser.parse_args()
    hps = HParams(
        nhidden=args.hidden_size,
        nembd=64,
        nbatch=args.batch_size,
        nstates=2,
        nvocab=256,
        out_wn=False,
        rnn_wn=True,
        rnn_type='mlstm',
        embd_wn=True,
    )

    X = tf.placeholder(tf.int32, [None, None])
    Y = tf.placeholder(tf.int32, [None, None])
    cells, states, logits = model(hps, X, reuse=False)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=Y)
    mean_loss = tf.reduce_mean(loss)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(mean_loss)
    loss = tf.reduce_sum(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run(session=sess)

    # load some data
    x = load_training_data(path=args.path, timesteps=args.timesteps, max_examples=args.max_examples)

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        for i in range(0, len(x), args.batch_size):
            batch_x = x[i:i + args.batch_size, 0:-1]
            batch_y = x[i:i + args.batch_size, 1:]
            _, batch_cost = sess.run((train_op, loss), {X: batch_x, Y: batch_y})
            epoch_loss += batch_cost
        t1 = time.time()
        print("%.3f\t%.3f" % (t1 - t0, epoch_loss))


if __name__ == "__main__":
    main()

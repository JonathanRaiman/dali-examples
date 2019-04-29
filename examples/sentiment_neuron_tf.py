import argparse
import tensorflow as tf
import numpy as np
import pytreebank
import time
import tqdm
from os.path import realpath, join, dirname

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


def preprocess(text, front_pad='\n ', end_pad=' '):
    text = text.replace('\n', ' ').strip()
    text = front_pad + text + end_pad
    text = text.encode()
    return text


def batchify(sentences):
    max_len = max(map(len, sentences))
    data = np.zeros((len(sentences), max_len + 1), dtype=np.int32)
    mask = np.zeros((len(sentences), max_len), dtype=np.float32)
    for idx, sent in enumerate(sentences):
        data[idx, 1:len(sent) + 1] = sent
        mask[idx, :len(sent)] = 1.0
    x, y, mask = data[:, :-1], data[:, 1:], mask
    return x, y, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
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
    # params = [np.load('model/%d.npy' % i) for i in range(15)]
    # params[2] = np.concatenate(params[2:6], axis=1)
    # params[3:6] = []

    X = tf.placeholder(tf.int32, [None, None])
    Y = tf.placeholder(tf.int32, [None, None])
    mask = tf.placeholder(tf.float32, [None, None])
    cells, states, logits = model(hps, X, reuse=False)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=Y)
    loss = loss * mask
    mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(mean_loss)
    loss = tf.reduce_sum(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run(session=sess)

    # load some data
    loaded_dataset = pytreebank.load_sst(join(DATA_DIR, "sst"))
    # labels = np.array([label for label, _ in text_data])
    text = np.array([list(preprocess(ex.to_lines()[0])) for ex in loaded_dataset['train']])
    batches_per_epoch = int(np.ceil(len(text) / args.batch_size))

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        for i in tqdm.tqdm(range(batches_per_epoch)):
            batch_indices = np.random.choice(len(text), size=args.batch_size)
            x, y, batch_mask = batchify(text[batch_indices])
            _, batch_cost = sess.run((train_op, loss), {X: x, Y: y, mask: batch_mask})
            epoch_loss += batch_cost
        t1 = time.time()
        print("%.3f\t%.3f" % (t1 - t0, epoch_loss))


if __name__ == "__main__":
    main()

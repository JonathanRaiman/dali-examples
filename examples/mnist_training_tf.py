from collections import namedtuple
import argparse
import tensorflow as tf
import numpy as np
from os.path import realpath, join, dirname
import time
from argparse_utils import add_bool_flag
from tensorflow.contrib.compiler import xla

CNNModel = namedtuple("CNNModel", "input_x input_y keep_prob output loss train_op")
SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


def build_model_fn(x, y, keep_prob):
    out = tf.reshape(x, [tf.shape(x)[0], 28, 28, 1])
    out = tf.contrib.layers.conv2d(out, 64, (5, 5), scope="conv1", padding="SAME", data_format="NHWC")
    out = tf.contrib.layers.max_pool2d(out, (2, 2), scope="pool1", padding="VALID", data_format="NHWC")
    out = tf.contrib.layers.conv2d(out, 64, (5, 5), scope="conv2", padding="SAME", data_format="NHWC")
    out = tf.contrib.layers.max_pool2d(out, (2, 2), scope="pool2", padding="VALID", data_format="NHWC")
    out = tf.reshape(out, [tf.shape(x)[0], 7 * 7 * 64])
    out = tf.nn.relu(out)
    out = tf.contrib.layers.fully_connected(out, 1024, activation_fn=tf.nn.relu, scope="fc1")
    out = tf.contrib.layers.dropout(out, keep_prob=keep_prob)
    out = tf.contrib.layers.fully_connected(out, 10, activation_fn=None, scope="fc2")
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(tf.reduce_mean(loss))

    with tf.control_dependencies([train_op]):
        loss_train_op = tf.identity(loss)
    return loss_train_op, loss, out


def build_model(use_xla):
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32, [])

    if use_xla:
        (train_op, loss, out) = xla.compile(computation=build_model_fn, inputs=(x, y, keep_prob))
    else:
        train_op, loss, out = build_model_fn(x, y, keep_prob)

    return CNNModel(x, y, keep_prob, out, tf.reduce_sum(loss), train_op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    add_bool_flag(parser, "xla", False)

    args = parser.parse_args()
    model = build_model(use_xla=args.xla)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    X = np.load(join(DATA_DIR, "mnist", "train_x.npy"))
    Y = np.load(join(DATA_DIR, "mnist", "train_y.npy"))
    batch_size = args.batch_size
    print(len(X))
    for iteration in range(args.epochs):
        t0 = time.time()
        for i in range(0, len(X), batch_size):
            _, batch_loss = sess.run((model.train_op, model.loss),
                                     {model.input_x: X[i:i + batch_size],
                                      model.input_y: Y[i:i + batch_size],
                                      model.keep_prob: 0.5})
        t1 = time.time()
        print("%.3f" % (t1 - t0))


if __name__ == "__main__":
    main()

from collections import namedtuple
import argparse
import tensorflow as tf
import numpy as np
from os.path import realpath, join, dirname
import time
CNNModel = namedtuple("CNNModel", "input_x input_y keep_prob output loss train_op")
SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


def build_model():
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32, [])
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
    return CNNModel(x, y, keep_prob, out, tf.reduce_sum(loss), train_op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    args = parser.parse_args()
    model = build_model()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    X = np.load(join(DATA_DIR, "mnist", "train_x.npy"))
    Y = np.load(join(DATA_DIR, "mnist", "train_y.npy"))
    batch_size = args.batch_size
    for iteration in range(10):
        t0 = time.time()
        for i in range(0, len(X), batch_size):
            _, batch_loss = sess.run((model.train_op, model.loss),
                                     {model.input_x: X[i:i + batch_size],
                                      model.input_y: Y[i:i + batch_size],
                                      model.keep_prob: 0.5})
        t1 = time.time()
        print("Iteration %d, Elapsed %.3f" % (iteration, t1 - t0))


if __name__ == "__main__":
    main()

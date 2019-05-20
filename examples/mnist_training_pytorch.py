from collections import namedtuple
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from os.path import realpath, join, dirname
import time
CNNModel = namedtuple("CNNModel", "input_x input_y keep_prob output loss train_op")
SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


def pad_same(in_dim, ks, stride, dilation=1):
    """
    Refernces:
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
    """
    assert stride > 0
    assert dilation >= 1
    effective_ks = (ks - 1) * dilation + 1
    out_dim = (in_dim + stride - 1) // stride
    p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

    padding_before = p // 2
    padding_after = p - padding_before
    return padding_before, padding_after


class Conv2dPaddingSame(nn.Module):
    def __init__(self, input_size, filters, kernel_size):
        self.W = nn.Parameter([filters, input_size, kernel_size, kernel_size])
        self.b = nn.Parameter([filters])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 64, 5, padding=pad_same(28, 5, 1))
        self.layer2 = nn.Conv2d(64, 64, 5, padding=pad_same(14, 5, 1))
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)
        self.cross_ent_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        out = x.reshape([x.shape[0], 1, 28, 28])
        out = torch.relu(self.layer1(out))
        out = F.max_pool2d(out, 2)
        out = torch.relu(self.layer2(out))
        out = F.max_pool2d(out, 2)
        out = torch.relu(torch.reshape(out, [out.shape[0], 7 * 7 * 64]))
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.cross_ent_loss(out, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    X = np.load(join(DATA_DIR, "mnist", "train_x.npy"))
    Y = np.load(join(DATA_DIR, "mnist", "train_y.npy"))
    batch_size = args.batch_size
    print(len(X))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for iteration in range(args.epochs):
        t0 = time.time()
        for i in range(0, len(X), batch_size):
            batch_x = torch.FloatTensor(X[i:i + batch_size])
            batch_y = torch.LongTensor(Y[i:i + batch_size])
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            batch_loss = model(batch_x, batch_y)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        t1 = time.time()
        print("%.3f" % (t1 - t0))


if __name__ == "__main__":
    main()

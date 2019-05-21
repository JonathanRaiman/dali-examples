import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import realpath, join, dirname
from sentiment_neuron_tf import load_training_data, HParams

SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")


class Model(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.prediction = nn.Linear(hps.nhidden, hps.nvocab)
        self.cross_ent_loss = nn.CrossEntropyLoss()
        self.embed = nn.Embedding(hps.nvocab, hps.nembd, padding_idx=0)

        nin = hps.nembd
        hidden_size = hps.nhidden

        self.wx = nn.Parameter(torch.randn([nin, hidden_size * 4]))
        self.wh = nn.Parameter(torch.randn([hidden_size, hidden_size * 4]))
        self.wmx = nn.Parameter(torch.randn([nin, hidden_size]))
        self.wmh = nn.Parameter(torch.randn([hidden_size, hidden_size]))
        self.b = nn.Parameter(torch.randn([hidden_size * 4]))
        self.wn = hps.rnn_wn
        self.hidden_size = hidden_size
        if hps.rnn_wn:
            self.gx = nn.Parameter(torch.randn([hidden_size * 4]))
            self.gh = nn.Parameter(torch.randn([hidden_size * 4]))
            self.gmx = nn.Parameter(torch.randn([hidden_size]))
            self.gmh = nn.Parameter(torch.randn([hidden_size]))

    def forward(self, xs, ys):
        xs = self.embed(xs)

        if self.wn:
            wx = F.normalize(self.wx, dim=0, p=2) * self.gx
            wh = F.normalize(self.wh, dim=0, p=2) * self.gh
            wmx = F.normalize(self.wmx, dim=0, p=2) * self.gmx
            wmh = F.normalize(self.wmh, dim=0, p=2) * self.gmh
        else:
            wx = self.wx
            wh = self.wh
            wmx = self.wmx
            wmh = self.wmh

        hs = []
        prev_h = torch.zeros((xs.shape[0], self.hidden_size))
        prev_c = torch.zeros((xs.shape[0], self.hidden_size))
        if torch.cuda.is_available():
            prev_h = prev_h.cuda()
            prev_c = prev_c.cuda()
        for step in range(xs.shape[1]):
            x = xs[:, step, :]
            m = torch.matmul(x, wmx) * torch.matmul(prev_h, wmh)
            z = torch.matmul(x, wx) + torch.matmul(m, wh) + self.b
            i, f, o, u = torch.chunk(z, 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            u = torch.tanh(u)
            c = f * prev_c + i * u
            h = o * torch.tanh(c)
            hs.append(h)
            prev_c = c
            prev_h = h
        hs = torch.stack(hs, dim=1)
        logits = self.prediction(hs)
        return self.cross_ent_loss(torch.reshape(logits, [logits.shape[0] * logits.shape[1], logits.shape[2]]),
                                   torch.reshape(ys, [ys.shape[0] * ys.shape[1]]))


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
    model = Model(hps)
    if torch.cuda.is_available():
        model.cuda()

    # load some data
    x = load_training_data(path=args.path, timesteps=args.timesteps, max_examples=args.max_examples)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        for i in range(0, len(x), args.batch_size):
            batch_x = torch.LongTensor(x[i:i + args.batch_size, 0:-1])
            batch_y = torch.LongTensor(x[i:i + args.batch_size, 1:])
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            batch_cost = model(batch_x, batch_y)
            model.zero_grad()
            batch_cost.backward()
            epoch_loss += float(batch_cost)
            optimizer.step()
        t1 = time.time()
        print("%.3f\t%.3f" % (t1 - t0, epoch_loss))


if __name__ == "__main__":
    main()

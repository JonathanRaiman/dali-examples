import numpy as np
from sklearn.datasets import fetch_mldata

from os.path import join, dirname, realpath, exists
from os import stat

SCRIPT_DIR = dirname(realpath(__file__))

def main():
    files = [
        join(SCRIPT_DIR, "train_x.npy"),
        join(SCRIPT_DIR, "train_y.npy"),
        join(SCRIPT_DIR, "validate_x.npy"),
        join(SCRIPT_DIR, "validate_y.npy"),
        join(SCRIPT_DIR, "test_x.npy"),
        join(SCRIPT_DIR, "test_y.npy")
    ]
    if all([exists(fname) and stat(fname).st_size > 100 for fname in files]):
        print("Already downloaded. Skipping")
    else:
        mnist = fetch_mldata('MNIST original')
        np.random.seed(1234)

        data = mnist.data
        target = mnist.target
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        data = data[indices]
        target = target[indices]

        train_x, train_y = (data[:-10000].astype(np.float32) / 255.0).astype(np.float32), target[:-10000].astype(np.int32)
        test_x, test_y = (data[-10000:].astype(np.float32) / 255.0).astype(np.float32), target[-10000:].astype(np.int32)

        np.save(join(SCRIPT_DIR, "train_x.npy"), train_x[:int(0.9 * train_x.shape[0])])
        np.save(join(SCRIPT_DIR, "train_y.npy"), train_y[:int(0.9 * train_y.shape[0])])
        np.save(join(SCRIPT_DIR, "validate_x.npy"), train_x[int(0.9 * train_x.shape[0]):])
        np.save(join(SCRIPT_DIR, "validate_y.npy"), train_y[int(0.9 * train_y.shape[0]):])

        np.save(join(SCRIPT_DIR, "test_x.npy"), test_x)
        np.save(join(SCRIPT_DIR, "test_y.npy"), test_y)
        print("Done.")

if __name__ == "__main__":
    main()

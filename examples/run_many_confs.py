import subprocess
import time
import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    my_env = os.environ.copy()
    confs = []
    for i in [0.5, 0.9, 1.0, 1.1, 3.0]:
        for j in [0.5, 0.9, 1.0, 1.1, 3.0]:
            for k in [0.5, 0.9, 1.0, 1.1, 3.0]:
                confs.append((i, j, k))

    for index, (i, j, k) in enumerate(confs):
        print("{}/{}".format(index, len(confs)))
        with open(args.output + "{}_{}_{}.txt".format(i, j, k), "wb") as fout:
            my_env["DALI_JIT_ALWAYS_RECOMPILE"] = "true"
            my_env["DALI_JIT_THREAD_WEIGHT"] = "{}".format(i)
            my_env["DALI_JIT_BLOCK_WEIGHT"] = "{}".format(j)
            my_env["DALI_JIT_THREADBLOCK_WEIGHT"] = "{}".format(k)
            out = subprocess.check_output("/home/jonathanraiman/Coding/dali-examples/build/mnist_training --device 0 --use_cudnn --use_jit_fusion --epochs 3".split(),
                env=my_env)
            fout.write(out)


if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
try:
    from tensor2tensor.data_generators import lm1b
    from tensor2tensor.data_generators.generator_utils import tfrecord_iterator_for_problem
except ImportError:
    print("To generate this data you must install tensor2tensor:\n"
          "pip install git+https://github.com/tensorflow/tensor2tensor")
    raise


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "~/t2t_data", type=str)
    parser.add_argument("--tmp_dir", "/tmp/t2t_data", type=str)
    args = parser.parse_args()
    gen = lm1b.LanguagemodelLm1b32k()
    data_dir = os.expanddir(args.data_dir)
    gen.generate_data(data_dir, args.tmp_dir)
    examples = []
    batch_size = 1024
    batches = 0
    for ex in tfrecord_iterator_for_problem(gen, data_dir):
        examples.append(ex["targets"].values)
        if len(examples) == batch_size:
            max_len = max(map(len, examples))
            batch = np.zeros((len(examples), max_len), dtype=np.int32)
            batch.fill(-1)
            for idx, ex in enumerate(examples):
                batch[idx, :len(ex)] = ex
            np.save(os.path.join(SCRIPT_DIR, "batch{}.npy".format(batches)), batch)
            batches += 1
            examples.clear()
            if batches % 100 == 0:
                print("batches {}".format(batches))


if __name__ == "__main__":
    main()

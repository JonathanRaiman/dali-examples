import os
import sys
import subprocess
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if sys.version_info >= (3, 3):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz"


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def main():
    path = os.path.join(SCRIPT_DIR, "reviews_Movies_and_TV_5.json.gz")
    final_path = os.path.join(SCRIPT_DIR, "reviews_Movies_and_TV_5.json")

    if os.path.exists(final_path) and os.stat(final_path).st_size > 1024 * 1024:
        print("File already downloaded and decompressed, done.")
        return

    if os.path.exists(path) and os.stat(path).st_size > 1024 * 1024:
        print("File already downloaded")
    else:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="download") as t:
            urlretrieve(URL, path, t.update_to)

    print("Decompressing file...")
    subprocess.check_call(["gzip", "-d", path], cwd=SCRIPT_DIR)
    print("Done")
    if os.path.exists(path) and os.stat(path).st_size > 1024 * 1024:
        print("Removing compressed file")
        # remove the compressed copy in favor of new copy
        os.remove(path)
        print("Done")

    assert os.path.exists(final_path) and os.stat(final_path).st_size > 1024 * 1024, "Did not get the final file."


if __name__ == "__main__":
    main()

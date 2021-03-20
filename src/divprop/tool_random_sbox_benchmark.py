'''
06:57:46.970 DEBUG __main__:HeavyPeeks: run #3089 fset frozenset({3, 4, 30, 23}) inv? True
06:57:54.573 INFO __main__:HeavyPeeks: computed divcore n=32 in 3089 bit-ANF calls, stat 1:32 2:490 3:998 4:33 1:32 2:489 3:990 4:25, size 7152
06:57:54.621 INFO __main__: computed divcore: 7152 elements

real    417m56.201s
user    402m38.419s
sys 9m6.920s
'''

import os
import gc
import subprocess
import pickle
import argparse

from binteger import Bin

from divprop.subsets import DenseSet, Sbox, Sbox32
from divprop.divcore import DivCore, SboxPeekANFs

import divprop.logs as logging

log = logging.getLogger(f"{__name__}:RandomSboxBenchmark")


class HeavyPeeks(SboxPeekANFs):
    log = logging.getLogger(f"{__name__}:HeavyPeeks")

    def __init__(self, n, fws, bks, cache_dir=None, memorize=False):
        self.n = int(n)
        self.cache_dir = cache_dir
        if memorize:
            self.log.info("loading forward coordinates to memory")
            self.fws = [DenseSet.load_from_file(f) for f in fws]
            self.log.info("loading backward coordinates to memory")
            self.bks = [DenseSet.load_from_file(f) for f in bks]
            self.log.info("loaded to memory done")
        else:
            self.fws = fws
            self.bks = bks
        self.memorize = memorize

    def get_coord(self, i, inverse):
        lst = self.bks if inverse else self.fws
        if self.memorize:
            return lst[i]
        else:
            return DenseSet.load_from_file(lst[i])

    def get_product(self, mask, inverse):
        cur = DenseSet(self.n)
        cur.fill()
        for i in Bin(mask, self.n).support():
            cur &= self.get_coord(i, inverse)
        return cur

    def run_mask(self, mask, inverse=False):
        if self.cache_dir is not None:
            str_mask = f"{mask:x}".zfill((self.n + 3) // 4)
            filename = os.path.join(
                self.cache_dir, f"{str_mask}_{['fw','bk'][inverse]}"
            )
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    try:
                        return pickle.load(f)
                    except Exception as err:
                        log.warning(f"cache error: file {filename} err {err}")
                        pass

        ret = super().run_mask(mask, inverse)

        if self.cache_dir is not None:
            with open(filename, "wb") as f:
                pickle.dump(ret, f)

        return ret


def tool_RandomSboxBenchmark():
    global log

    parser = argparse.ArgumentParser(
        description="Generate division core of a random S-box for benchmark."
    )

    parser.add_argument(
        "n", type=int,
        help="bit size of the S-box",
    )
    parser.add_argument(
        "-l", "--large", action="store_true",
        help="Large S-box (use extensive caching and files)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="divcore_random",
        help="Base directory for files (logs, cache, divcore, etc.)",
    )

    args = parser.parse_args()

    n = args.n

    path = os.path.join(args.output, f"{n:02d}")
    os.makedirs(path, exist_ok=True)

    logging.addFileHandler(f"{path}/log")
    logging.setup(level="DEBUG")

    log.info(f"{args}")

    if args.large:
        run_large(n, path)
    else:
        run_small(n, path)


def run_large(n, path):
    filename = f"{path}/fw.sbox"
    ifilename = f"{path}/bk.sbox"
    last_filename = f"{path}/bk{n-1}.set"

    if not os.path.isfile(last_filename):
        log.info(f"generating {n}-bit S-box...")
        sbox = Sbox32.GEN_random_permutation(n, 2021)
        log.info(f"{sbox}")

        log.info(f"saving to {filename} ...")
        sbox.save_to_file(filename)

        log.info("hashing...")
        h = subprocess.check_output(["sha256sum", filename]).split()[0]
        log.info(f"sha256sum: {h}")

        log.info("splitting into coordinates...")
        for i in range(n):
            coord = sbox.coordinate(i)
            coord.save_to_file(f"{path}/fw{i}.set")
            log.info(f"coord {i}/{n} saved")

        log.info("inverting...")
        # somehow ~sbox caused extra (temporary) instance
        isbox = sbox
        del sbox
        isbox.invert_in_place()
        gc.collect()

        log.info(f"saving to {ifilename} ...")
        isbox.save_to_file(ifilename)

        log.info("hashing...")
        h = subprocess.check_output(["sha256sum", ifilename]).split()[0]
        log.info(f"sha256sum: {h}")

        log.info("splitting into coordinates...")
        for i in range(n):
            coord = isbox.coordinate(i)
            coord.save_to_file(f"{path}/bk{i}.set")
            log.info(f"coord {i}/{n} saved")

    log.info("heavy peeks")
    fws = [f"{path}/fw{i}.set" for i in range(n)]
    bks = [f"{path}/bk{i}.set" for i in range(n)]

    cache_dir = f"{path}/cache/"
    os.makedirs(cache_dir, exist_ok=True)
    pa = HeavyPeeks(n, fws, bks, cache_dir=cache_dir, memorize=True)
    res = sorted(pa.compute())

    divcore_file = f"{path}/divcore.txt"
    log.info(f"divcore: {len(res)} elements, saving to {divcore_file} ...")

    with open(divcore_file, "w") as f:
        print(len(res), file=f)
        for uv in res:
            print(int(uv), file=f, end=" ")


def run_small(n, path):
    assert n < 24, "are you crazy?"

    log.info(f"generating {n}-bit S-box...")
    sbox = Sbox32.GEN_random_permutation(n, 1)
    log.info(f"{sbox}")

    filename = f"{path}/fw.sbox"
    log.info(f"saving to {filename} ...")
    sbox.save_to_file(filename)

    log.info("hashing...")
    h = subprocess.check_output(["sha256sum", filename])
    log.info(f"sha256sum: {h}")

    log.info("computing division core...")
    pa = SboxPeekANFs(sbox)
    log.info("sorting...")
    res = sorted(pa.compute())

    divcore_file = f"{path}/divcore.txt"
    log.info(f"divcore: {len(res)} elements, saving to {divcore_file} ...")

    with open(divcore_file, "w") as f:
        print(len(res), file=f)
        for uv in res:
            print(int(uv), file=f, end=" ")

    if n <= 16:
        ans = sorted(DivCore.from_sbox(sbox).to_Bins())
        assert res == ans


if __name__ == '__main__':
    tool_RandomSboxBenchmark()

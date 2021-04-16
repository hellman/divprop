import time

from random import shuffle, random
from itertools import combinations

from collections import Counter, defaultdict

from binteger import Bin

from subsets import SparseSet

from divprop.logs import logging

from .LearnModule import LearnModule


class LevelLearn(LearnModule):
    log = logging.getLogger(f"{__name__}")

    def __init__(self, levels_lower=0, levels_upper=0):
        self.levels_lower = int(levels_lower)
        self.levels_upper = int(levels_upper)

    def _learn(self):
        if self.levels_lower:
            self.learn_lower(up_to=self.levels_lower - 1)
        if self.levels_upper:
            self.learn_upper(down_to=self.n - self.levels_upper + 1)

    def learn_lower(self, up_to):
        cache = self.system.oracle._lower_cache

        if cache.range is None:
            current = -1
        else:
            assert cache.range[0] == 0
            current = cache.range[1]

        if current < 0:
            vec = self.vec_empty

            is_lower, meta = self.oracle(vec)
            if is_lower:
                self.system.meta[vec] = meta
                cache.add(vec, meta)
            cache.set_range(0, 0)
            current = 0

        if not cache.has(self.vec_empty):
            self.log.warning("0-vector is not in lower set, trivial set")
            return

        for l in range(current + 1, up_to + 1):
            self.log.info(f"generating support, height={l}/{up_to}")

            n_good = 0
            n_total = 0

            # cache stores only lower vectors
            # only check new vectors that are compatible with lowers
            # can be checked by counting refs
            to_check = {}
            for prev in cache.iter_weight(l - 1):
                for up in prev.neibs_up(n=self.N):
                    to_check.setdefault(up, 0)
                    to_check[up] += 1

            for vec, cnt in to_check.items():
                assert len(vec) == l
                if cnt != l:
                    continue

                n_total += 1

                is_lower, meta = self.oracle(vec)
                if is_lower:
                    self.system.meta[vec] = meta
                    cache.add(vec, meta)
                    n_good += 1
                else:
                    # print("upper", vec)
                    self.system.add_upper(vec, meta=meta, is_prime=True)

            self.log.info(
                f"generated support, height={l}/{up_to}: "
                f"lower {n_good}/{n_total} compatible "
                f"(frac. {(n_good+1)/(n_total+1):.3f})"
            )
            cache.set_range(0, l)

            if n_good == 0:
                self.log.warning(f"exhausted lower at level {l}/{up_to}")
                break

    def learn_upper(self, down_to):
        raise NotImplementedError()


class UnknownMeta:
    pass


class LevelCache:
    def __init__(self):
        self.cache = []
        self.meta = {}
        self.range = None

    def add(self, vec, meta=None):
        while len(self.cache) <= len(vec):
            self.cache.append({})
        if meta is not None:
            self.meta[vec] = meta
        self.cache[len(vec)][vec] = meta

    def has(self, vec):
        if self.range is None or not self.range[0] <= len(vec) <= self.range[1]:
            # can not check
            return
        return vec in self.cache[len(vec)]

    def set_range(self, start, end):
        self.range = start, end

    def iter_weight(self, weight):
        if weight >= len(self.cache):
            return iter(())
        return iter(self.cache[weight])

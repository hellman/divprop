import os
import pickle

from itertools import chain
from collections import Counter
from queue import Queue

from divprop.subsets import SparseSet
from divprop.logs import logging

from .LevelLearn import LevelCache


class ExtraPrec:
    def reduce(self, vec: SparseSet):
        return vec

    def expand(self, vec: SparseSet):
        return vec


class LowerSetLearn:
    DATA_VERSION = 1
    log = logging.getLogger(f"{__name__}:LowerSetLearn")

    def __init__(
            self,
            n: int,
            file: str = None,
            oracle: callable = None,
            extra_prec: ExtraPrec = None,
        ):

        self.n = int(n)

        self.file = file
        self.oracle = oracle
        self.extra_prec = extra_prec

        self._lower = set()
        self._upper = set()

        self._lower_cache = LevelCache()
        self._upper_cache = LevelCache()

        self.meta = {}  # info per elements of lower/upper

        self.saved = False
        if self.file and os.path.exists(self.file):
            self.load()

    def save(self):
        if self.file and not self.saved:
            try:
                self.save_to_file(self.file)
                self.saved = True
            except KeyboardInterrupt:
                self.log.error(
                    "interrupted saving! trying again, please be patient"
                )
                self.save_to_file(self.file)
                self.saved = True
                raise
        self.log_info()

    def load(self):
        if self.file:
            self.load_from_file(self.file)
            self.log_info()
            self.saved = True

    def load_from_file(self, filename):
        prevn = self.n
        with open(filename, "rb") as f:
            data = pickle.load(f)
        (
            version,
            self._lower, self._lower_cache,
            self._upper, self._upper_cache,
            self.meta, self.n,
        ) = data
        assert version == self.DATA_VERSION, "system format updated?"
        assert self.n == prevn
        self.log.info(f"loaded state from file {filename}")

    def save_to_file(self, filename):
        data = (
            self.DATA_VERSION,
            self._lower, self._lower_cache,
            self._upper, self._upper_cache,
            self.meta, self.n,
        )
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        self.log.info(f"saved state to file {filename}")

    def log_info(self):
        for (name, s) in [
            ("lower", self._lower),
            ("upper", self._upper),
        ]:
            freq = Counter(len(v) for v in s)
            freqstr = " ".join(
                f"{sz}:{cnt}" for sz, cnt in sorted(freq.items())
            )
            self.log.info(f"  {name} {len(s)}: {freqstr}")

    def is_known_lower(self, vec):
        return vec in self._lower or self._lower_cache.has(vec)

    def is_known_upper(self, vec):
        return vec in self._upper or self._upper_cache.has(vec)

    def is_prime_lower(self, vec):
        return vec in self._lower  # or vec in self._lower_cache

    def is_prime_upper(self, vec):
        return vec in self._upper  # or vec in self._upper_cache

    def add_lower(self, vec, meta=None, is_prime=False):
        assert isinstance(vec, SparseSet)

        if self.extra_prec:
            vec = self.extra_prec.expand(vec)

        # in case of interrupt, consistency is kept
        if not self.is_known_lower(vec):
            self.saved = False

            if meta is not None:
                self.meta[vec] = meta

            self._lower.add(vec)

    def add_upper(self, vec, meta=None, is_prime=False):
        assert isinstance(vec, SparseSet)

        if self.extra_prec:
            vec = self.extra_prec.reduce(vec)

        # in case of interrupt, consistency is kept
        if not self.is_known_upper(vec):
            self.saved = False

            if meta is not None:
                self.meta[vec] = meta

            self._upper.add(vec)

    def iter_lower(self):
        return iter(self._lower)

    def iter_upper(self):
        return iter(self._upper)

    def n_lower(self):
        return len(self._lower)

    def n_upper(self):
        return len(self._upper)


class ExtraPrec_LowerSet(ExtraPrec):
    def __init__(self, int2point: list, point2int: map):
        self.int2point = int2point
        self.point2int = point2int

    def reduce(self, vec: SparseSet):
        """MaxSet"""
        res = set()
        qs = [self.int2point[i] for i in vec]
        assert all(isinstance(q, SparseSet) for q in qs)
        for pi, p in enumerate(qs):
            if not any(p < q for q in qs):
                res.add(self.point2int[p])
        return SparseSet(res)

    def expand(self, vec: SparseSet):
        """LowerSet"""
        res = set()
        qs = [self.int2point[i] for i in vec]
        assert all(isinstance(q, SparseSet) for q in qs)

        # simple BFS-like algorithm
        pq = Queue()
        for q in qs:
            pq.put(q)

        visited = set(qs)
        while pq.qsize():
            q = pq.get()
            for sub in q.neibs_down():
                if sub not in visited:
                    visited.add(sub)
                    pq.put(sub)
        res = []
        for q in visited:
            if q in self.point2int:
                res.append(self.point2int[q])
        return SparseSet(res)

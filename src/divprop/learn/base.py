import time
import logging
import pickle

from tqdm import tqdm
from random import choice, shuffle, randrange
from itertools import combinations

from collections import Counter, defaultdict
from queue import PriorityQueue, Queue

from binteger import Bin

from divprop.milp import MILP
from divprop.subsets import SparseSet
from divprop.logs import logging


class ExtraPrec:
    def reduce(self, vec: SparseSet):
        return vec

    def expand(self, vec: SparseSet):
        return vec


class LowerSetLearn:
    log = logging.getLogger(f"{__name__}:LowerSetLearn")

    def __init__(
            self,
            n: int,
            file: str = None,
            oracle: callable = None,
            extra_prec: ExtraPrec = None,
        ):

        self.n = int(n)

        self.oracle = None
        self.extra_prec = extra_prec
        self.file = file

        self.lower = set()
        self.upper = set()
        self.lower_is_prime = True
        self.upper_is_prime = True
        self.meta = {}  # info per elements of lower/upper

    def set_oracle(self, oracle: callable):
        self.oracle = oracle

    def save(self):
        if self.file:
            try:
                self.save_to_file(self.file)
            except KeyboardInterrupt:
                self.log.error(
                    "interrupted saving! trying again, please be patient"
                )
                self.save_to_file(self.file)
                raise

    def load(self):
        if self.file:
            self.load_from_file(self.file)

    def load_from_file(self, filename):
        prevn = self.n
        with open(filename, "rb") as f:
            data = pickle.load(f)
        (
            self.lower, self.upper, self.meta, self.n,
            self.lower_is_prime, self.upper_is_prime,
        ) = data
        assert self.n == prevn
        self.log.info(f"loaded state from file {filename}")
        self.log_info()

    def save_to_file(self, filename):
        data = (
            self.lower, self.upper, self.meta, self.n,
            self.lower_is_prime, self.upper_is_prime,
        )
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        self.log.info(f"saved state to file {filename}")
        self.log_info()

    def log_info(self):
        for (name, s) in [
            ("lower", self.lower),
            ("upper", self.upper),
        ]:
            freq = Counter(len(v) for v in s)
            freqstr = " ".join(
                f"{sz}:{cnt}" for sz, cnt in sorted(freq.items())
            )
            self.log.info(f"  {name} {len(s)}: {freqstr}")

    def add_lower(self, vec, info=None, is_prime=False):
        assert isinstance(vec, SparseSet)
        if not is_prime:
            self.lower_is_prime = False

        if self.extra_prec:
            vec = self.extra_prec.expand(vec)

        if vec not in self.lower:
            self.lower.add(vec)
            if info is not None:
                self.meta[vec] = info

    def add_upper(self, vec, info=None, is_prime=False):
        assert isinstance(vec, SparseSet)
        if not is_prime:
            self.upper_is_prime = False

        if self.extra_prec:
            vec = self.extra_prec.reduce(vec)

        if vec not in self.upper:
            self.upper.add(vec)
            if info is not None:
                self.meta[vec] = info


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

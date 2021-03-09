import logging
from random import randint, choice, sample
from collections import Counter, namedtuple
from math import gcd

from tqdm import tqdm
from binteger import Bin

from .base import (
    inner, satisfy,
    MixedIntegerLinearProgram,
    Polyhedron,
)
from .random_group_cut import RandomGroupCut
from .gem_cut import GemCut


log = logging.getLogger(__name__)


def notpoint(p):
    assert 0 <= min(p) <= max(p) <= 1
    return tuple(1 ^ v for v in p)


IneqInfo = namedtuple("SourcedIneq", ("ineq", "source", "state"))


class MonotoneInequalitiesPool:
    def __init__(self, points_good, points_bad, type_good=None):
        for p in points_bad:
            self.n = len(p)
            break

        assert type_good in ("lower", "upper")
        self.type_good = type_good
        if type_good == "upper":
            self.lo = sorted(map(tuple, points_bad))
            self.hi = sorted(map(tuple, points_good))
        else:
            self.lo = sorted(map(notpoint, points_good))
            self.hi = sorted(map(notpoint, points_bad))

        self.i2lo = list(self.lo)
        self.lo2i = {p: i for i, p in enumerate(self.i2lo)}
        self.N = len(self.lo)

        self.LSL = SparseLowerSetLearn()

        # initialize
        for pi, p in enumerate(self.i2lo):
            self.gen_basic_ineq(pi, p)

    # python impl goal: smallish s-boxes
    # + maybe OK random ineqs / pure hats

    def gen_basic_ineq(self, pi, p):
        fset = self.LSL.encode_fset((pi,))
        # does not belong to LowerSet({p})
        # <=>
        # sum coords with p_i = 0 >= 1
        ineq = tuple(1 if x == 0 else 0 for x in p) + (-1,)
        self.LSL.add_feasible(fset, ineq=ineq, source="basic")

    def gen_random_inequality(self, max_coef=100):
        lin = [randrange(max_coef+1) for _ in range(self.n)]
        ev_good = [inner(p, lin) for p in self.hi]
        fset = self.LSL.encode_fset(
            q for q in self.lo if inner(q, lin) < ev_good
        )
        # lin. comb. >= ev_good
        ineq = tuple(lin) + (-ev_good,)
        self.LSL.add_feasible(fset, ineq=ineq, source="random_ineq")

    def gen_all_hats(self):
        '''
        unordered_map<T, vector<T>> neibors;
        unordered_map<T, int> neibors_size;

        for (auto u: lb) {
            for (auto supu: neibs_up(u, n)) {
                neibors[supu].push_back(u);
            }
        }
        for (auto &p: neibors) {
            neibors_size[p.first] = p.second.size();
        }
        '''

    def try_extend_highest(self):
        '''
        extend

        extending top improves FEASIBLE
        but extending bot improves INFEASIBLE
        both are useful

        infeasible seems cutting off more checks
        BFS
        '''

    def try_extend_lowest(self):
        dunno


# TBD: base class interface, dense class for smallish N (<100? < 1000?)
class SparseLowerSetLearn:
    def __init__(self, N, oracle):
        self.N = int(N)
        self.coprimes = [
            i for i in range(1, self.N)
            if gcd(i, self.N) == 1
        ]
        self.oracle = oracle

        # TBD: optimization by hw?
        # { set of indexes of covered bad points (hi)
        #   :
        #   ineq, source, state }
        # state = int index of last unchecked bit up?
        self.feasible = {}
        self.feasible_open = set()

        # { set of indexes of infeasible to cover bad points }
        self.infeasible = set()

        self._order_sbox = sample(range(self.N), self.N)

    def encode_fset(self, fset):
        return frozenset(map(int, fset))

    def _hash(self, fset):
        mask = 2**64-1
        res = 0x9c994e7c9068e947
        for v in fset:
            res ^= v
            res *= 0xf7ace5e55fd1c1ad
            res &= mask
            res ^= res >> 17
            res &= mask
        return res

    def _get_real_index(self, h, i):
        i += h + 0x28a5e1f1
        i %= self.N
        i = self._order_sbox[i]
        i += h + 0xb5520e03
        i %= self.N
        i = self._order_sbox[i]
        i += h + 0xb12dcbaa
        i %= self.N
        return i

    def is_already_feasible(self, fset):
        # quick check
        if fset in self.feasible:
            return True
        # is in feasible lowerset?
        for fset2 in self.feasible:
            if fset <= fset2:
                return True
        return False

    def is_already_infeasible(self, fset):
        # quick check
        if fset in self.infeasible:
            return True
        # is in infeasible upperset?
        for fset2 in self.infeasible:
            if fset2 <= fset:
                return True
        return False

    def add_feasible(self, fset, ineq, source, check=True):
        if check and self.is_already_feasible(p):
            return
        # remove existing redundant
        self.feasible = {
            fset2: info2
            for fset2, info2 in self.feasible.items()
            if not (fset2 <= fset)
        }
        self.feasible_open = {
            fset2
            for fset2 in self.feasible_open
            if not (fset2 <= fset)
        }
        self.feasible[fset] = IneqInfo(
            ineq,
            source="basic",
            state=(self._hash(fset), 0)
        )

    def add_infeasible(self, fset, check=True):
        if check and self.is_already_infeasible(fset):
            return
        # remove existing redundant
        self.infeasible = {
            fset2
            for fset2 in self.infeasible
            if not (fset2 >= fset)
        }
        self.infeasible.add(fset)

    def get_next_unknown_neighbour(self, fset):
        assert fset in self.feasible_open
        h, i = self.feasible[fset].state
        while i < self.N:
            ii = self._get_real_index(h, i)
            i += 1
            if ii not in fset:
                fset2 = fset | {ii}
                if fset2 in self.infeasible:
                    continue
                if self.is_already_feasible(fset2):
                    continue
                if self.is_already_infeasible(fset2):
                    continue
                break
        else:
            assert 0, "no neighbours left? invariant broken"

        if i >= self.N:
            i = None
            self.feasible_open.remove(fset)

        self.feasible_open[fset] = self.feasible_open[fset].replace(
            state=(h, i))

        return fset2

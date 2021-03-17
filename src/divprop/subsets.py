"""
Manipulations on sets of binary vectors / subsets of some set.
"""

from itertools import combinations

from .libsubsets import *
from .libsubsets import DenseSet

from binteger import Bin


def QMC1(P, n=None):
    """
    See ToSC:BouCog20.
    Reformulation of Quine-McCluskey algorithm in two parts:
    1. finding all subsets of (P) of the form (a xor LowerSet(u))
       with maximal (u).
    2. finding a good / optimal covering of P with such subsets

    This method does part 1 in the framework of dense sets.

    Complexity: n 2^n |P|
    """
    if n is None:
        assert isinstance(P, DenseSet), "n must be given or P must be a DenseSet"
        n = P.n

    S = []
    for a in P:
        a = Bin(a, n).int

        # TBD: do in place and return orig P (dangereux?)
        X = P.copy()
        X.do_Not(a)
        X.do_Complement()
        X.do_UpperSet()
        X.do_Complement()

        # interesting experiment
        if 0:
            AU = DenseSet(n)
            nota = (2**n-1) ^ a
            AU.set(nota)
            AU.do_LowerSet()
            test1 = X.MaxSet() & AU
            test2 = (X & AU).MaxSet()
            print("A", Bin(a, n))
            if test1 != test2:
                print("X", X)
                for v in [Bin(v, n) for v in X]:
                    print(v)
                print("X & AU", X & AU)
                for v in [Bin(v, n) for v in X & AU]:
                    print(v)
                print("X maxset", X.MaxSet())
                for v in [Bin(v, n) for v in X.MaxSet()]:
                    print(v)
                print("X maxset & AU", test1)
                for v in [Bin(v, n) for v in test1]:
                    print(v)
                print("X & AU -> maxset", test2)
                for v in [Bin(v, n) for v in test2]:
                    print(v)
                quit()

        X.do_MaxSet()
        X.do_UnsetUp(a)
        for u in X:
            S.append((a, u))
    return S


def hw(v):
    return bin(v).count("1")


class WeightedSet:
    def __init__(self, spec, n):
        self.n = int(n)
        self.sets = [set() for i in range(self.n+1)]
        for v in spec:
            self.add(v)

    def add(self, v):
        self.sets[hw(v)].add(v)

    def iter_ge(self, w=0):
        for s in self.sets[w:]:
            yield from s

    def iter_le(self, w=None):
        if w is None:
            w = self.n
        for s in self.sets[:w+1]:
            yield from s

    def iter_wt(self, w):
        yield from self.sets[w]

    def __iter__(self):
        return iter(self.iter_ge())

    def __contains__(self, v):
        return v in self.sets[hw(v)]

    def remove(self, v):
        s = self.sets[hw(v)]
        return s.remove(v)

    def __len__(self):
        return sum(len(v) for v in self.sets)

    def do_MaxSet(self):
        w = max(w for w in range(self.n+1) if self.sets[w])
        # remove from w1 using w2
        for w2 in reversed(range(w+1)):
            for w1 in reversed(range(w2)):
                if not self.sets[w1]:
                    continue
                self.sets[w1] = {
                    v for v in self.sets[w1]
                    if not any(v & u == v for u in self.sets[w2])
                }


class WeightedFrozenSets:
    def __init__(self, n, spec=()):
        self.n = int(n)
        self.sets = [set() for i in range(self.n+1)]
        for v in spec:
            self.add(v)

    def add(self, v: frozenset):
        self.sets[len(v)].add(v)

    def iter_ge(self, w=0):
        for s in self.sets[w:]:
            yield from s

    def iter_le(self, w=None):
        if w is None:
            w = self.n
        for s in self.sets[:w+1]:
            yield from s

    def iter_wt(self, w):
        yield from self.sets[w]

    def __iter__(self):
        return iter(self.iter_ge())

    def __contains__(self, v):
        return v in self.sets[len(v)]

    def remove(self, v):
        return self.sets[len(v)].remove(v)

    def discard(self, v):
        return self.sets[len(v)].discard(v)

    def __len__(self):
        return sum(len(v) for v in self.sets)

    def __getitem__(self, w):
        return self.sets[w]

    def do_MaxSet(self):
        """naive, optimized by weights"""
        w = max(w for w in range(self.n+1) if self.sets[w])
        # remove from w1 using w2
        for w2 in reversed(range(w+1)):
            for w1 in reversed(range(w2)):
                if not self.sets[w1]:
                    continue
                self.sets[w1] = {
                    v for v in self.sets[w1]
                    if not any(v & u == v for u in self.sets[w2])
                }


class GrowingExtremeFrozen:
    """
    Invariant:
        extreme <= sets <= cache

    """
    def __init__(self, n, spec=(), disable_cache=False):
        self.n = int(n)
        self.sets = [set() for i in range(self.n+1)]
        if disable_cache:
            self.cache = None
        else:
            self.cache = WeightedFrozenSets(n=self.n)

        for v in spec:
            self.add(v)

    def add(self, v: frozenset):
        if self.cache and v in self.cache[len(v)]:
            return
        self.sets[len(v)].add(v)
        if self.cache:
            self.cache[len(v)].add(v)

    def iter_ge(self, w=0):
        for s in self.sets[w:]:
            yield from s

    def iter_le(self, w=None):
        if w is None:
            w = self.n
        if w < 0:
            return
        assert 0 <= w <= self.n
        for s in reversed(self.sets[:w+1]):
            yield from s

    def iter_wt(self, w):
        yield from self.sets[w]

    def __iter__(self):
        return iter(self.iter_ge())

    def __contains__(self, v):
        return self.contains(v, strict=False)

    def __len__(self):
        return sum(len(v) for v in self.sets)

    def to_Bins(self):
        for v in self:
            yield Bin(v, self.n)


class GrowingLowerFrozen(GrowingExtremeFrozen):
    # low_cache_level = -1
    # upper_descend_max = 10**4

    def contains(self, v, strict=False):
        """naive, optimized by weights"""
        assert not strict, "not impl"
        if self.cache and v in self.cache:
            return True

        for w in reversed(range(len(v), self.n+1)):
            for u in self.sets[w]:
                if u | v == u:
                    return True
        return False

    def do_MaxSet(self):
        """naive, optimized by weights"""
        # remove from w1 using w2
        for w2 in reversed(range(1, self.n+1)):
            if not self.sets[w2]:
                continue
            for w1 in range(w2):
                if not self.sets[w1]:
                    continue
                # rough binomial(w2, w1)
                nsub = w2**w1 if w1 < w2//2 else (w2-w1)**w1
                if nsub < len(self.sets[w1]):
                    # enum all subseqs
                    for u in self.sets[w2]:
                        for v in combinations(u, w1):
                            self.sets[w1].discard(frozenset(v))
                else:
                    self.sets[w1] = {
                        v for v in self.sets[w1]
                        if not any(v & u == v for u in self.sets[w2])
                    }


class GrowingUpperFrozen(GrowingExtremeFrozen):
    def contains(self, v, strict=False):
        """naive, optimized by weights"""
        assert not strict, "not impl"
        if self.cache and v in self.cache:
            return True

        for w in range(len(v)+1):
            for u in self.sets[w]:
                if u & v == u:
                    return True
        return False

    def do_MinSet(self):
        """naive, optimized by weights"""
        # remove from w1 using w2
        for w1 in range(1, self.n+1):
            todel = set()
            for v in self.sets[w1].copy():
                is_red = 0
                for w2 in range(w1):
                    nsub = w1**w2 if w2 < w1//2 else (w1-w2)**w2
                    full = frozenset(range(self.n))
                    if nsub < len(self.sets[w2]):
                        for u in combinations(v, w2):
                            if frozenset(u) in self.sets[w2]:
                                is_red = 1
                                break
                    else:
                        if any(v | u == v for u in self.sets[w2]):
                            is_red = 1

                    if is_red:
                        todel.add(v)
                        break
            self.sets[w1] -= todel
        return


        for w2 in range(0, self.n):
            if not self.sets[w2]:
                continue
            for w1 in range(w2+1, self.n+1):
                if not self.sets[w1]:
                    continue

                # rough binomial(nw2, nw1)
                nw1 = self.n - w1
                nw2 = self.n - w2
                nsup = nw2**nw1 if nw1 < nw2//2 else (nw2-nw1)**nw1
                full = frozenset(range(self.n))
                if nsup < len(self.sets[w1]):
                    # enum all superseqs
                    for u in self.sets[w2]:
                        nu = fset_complement(u, self.n)
                        for v in combinations(nu, nw1):
                            v = full - frozenset(v)
                            self.sets[w1].discard(v)
                else:
                    # quadratic algo
                    self.sets[w1] = {
                        v for v in self.sets[w1]
                        if not any(v | u == v for u in self.sets[w2])
                    }


class DynamicExtremeSet:
    def __init__(self, spec, n):
        self.set = set(spec)
        self.n = int(n)

    def to_DenseSet(self):
        return DenseSet(tuple(self.set), self.n)


class DynamicLowerSet(DynamicExtremeSet):
    def remove_upper_singleton(self, v: int):
        # complement of upperset({00 111 00})
        # is lowerset({11 011 11, 11 101 11, 11 110 11})
        # intersect with their union
        inds = support_int_le(v, self.n)

        prevset = self.set
        self.set = set()
        to_add = set()
        for u in prevset:
            # v <= u?
            if v & u != v:
                # no intersection, keep
                self.set.add(u)
            else:
                # intersection, split
                for i in inds:
                    to_add.add(u ^ (1 << i))

        # print("to_add", len(to_add))
        self._added_last = set()
        for u in to_add:
            if u not in self:
                self.set.add(u)
                self._added_last.add(u)

    def add_lower_singleton(self, v: int, check=True):
        if check and v in self:
            return False
        self.set = {u for u in self.set if not (u & v == u)}
        self.set.add(v)
        return True

    def remove_lower_singleton_extremes(self, v: int):
        """remove elements from the maxset that are <= v"""
        self.set = {u for u in self.set if not (u & v == u)}

    def __contains__(self, v: int):
        if v in self.set:
            return True
        for u in self.set:
            if v & u == v:  # v <= u
                return True
        return False


class DynamicUpperSet(DynamicExtremeSet):
    def remove_lower_singleton(self, v: int):
        inds = antisupport_int_le(v, self.n)

        prevset = self.set
        self.set = set()
        to_add = set()
        for u in prevset:
            # v >= u?
            if v | u != v:
                # no intersection, keep
                self.set.add(u)
            else:
                # intersection, split
                for i in inds:
                    to_add.add(u ^ (1 << i))

        # print("to_add", len(to_add))
        self._added_last = set()
        for u in to_add:
            if u not in self:
                self.set.add(u)
                self._added_last.add(u)

    def add_upper_singleton(self, v: int, check=True):
        if check and v in self:
            return False
        self.set = {u for u in self.set if not (u | v == u)}
        self.set.add(v)
        return True

    def remove_upper_singleton_extremes(self, v: int):
        """remove elements from the maxset that are >= v"""
        self.set = {u for u in self.set if not (u | v == u)}

    def __contains__(self, v: int):
        if v in self.set:
            return True
        for u in self.set:
            if v | u == v:  # v >= u
                return True
        return False


class OptDynamicUpperSet:
    def __init__(self, spec, n):
        self.set = set(spec)
        self.set = WeightedSet(spec, n)
        self.n = int(n)

    def to_DenseSet(self):
        return DenseSet(tuple(self.set), self.n)

    def remove_lower_singleton(self, v: int):
        inds = antisupport_int_le(v, self.n)

        self._added_last = set()
        for w in range(hw(v) + 1):
            to_split = set()
            for u in self.set.sets[w]:
                # v >= u?
                if v | u == v:
                    # intersection, split
                    to_split.add(u)

            for u in to_split:
                self.set.sets[w].remove(u)
                for i in inds:
                    uu = u ^ (1 << i)
                    if 0:
                        self.set.sets[w+1].add(uu)
                        self._added_last.add(uu)
                    else:
                        for vv in self.set.iter_le(w):
                            if vv & uu == vv:
                                break
                        else:
                            self.set.sets[w+1].add(uu)
                            self._added_last.add(uu)

            # for u in to_add:
            #     good = 1
            #     for ww in range(w):
            #         for v in self.set.sets[ww]:
            #             if v & u == v:
            #                 good = 0
            #                 break
            #         if not good:
            #             break
            #     if good or 1:
            #         self.set.sets[w+1].add(u)
            #         pass
            # print("to_add", len(to_add))
            # for u in to_add:
            #     if u not in self.set:
            #         newset.add(u)

    # def add_upper_singleton(self, v: int, check=True):
    #     if check and v in self:
    #         return False

    #     for w in range(hw(v)+1, self.N+1):
    #         todel = {u for u in self.set.sets[w] if u | v == u}
    #         self.set.sets[w] -= todel

    #     self.set.add(v)
    #     return True

    # def remove_upper_singleton_extremes(self, v: int):
    #     """remove elements from the maxset that are >= v"""
    #     self.set = {u for u in self.set if not (u | v == u)}

    def __contains__(self, v: int):
        if v in self.set:
            return True
        for u in self.set:
        # for u in self.set.iter_ge(hw(v) + 1):
            if v | u == v:  # v >= u
                return True
        return False


def not_tuple(p):
    assert 0 <= min(p) <= max(p) <= 1
    return tuple(1 ^ v for v in p)


def neibs_up_tuple(p):
    p = list(p)
    for i in range(len(p)):
        if p[i] == 0:
            p[i] = 1
            yield tuple(p)
            p[i] = 0


def neibs_up_int(v, n):
    for i in reversed(range(n)):
        if v & (1 << i) == 0:
            yield v | (1 << i)


def support_int_le(v, n):
    return [i for i in range(n) if v & (1 << i)]


def antisupport_int_le(v, n):
    return [i for i in range(n) if v & (1 << i) == 0]


def fset_complement(fset, n):
    return frozenset(i for i in range(n) if i not in fset)

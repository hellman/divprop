"""
Manipulations on sets of binary vectors / subsets of some set.
"""

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


class DynamicExtremeSet:
    def __init__(self, spec, n):
        self.set = set(spec)
        self.n = int(n)


class DynamicLowerSet(DynamicExtremeSet):
    def remove_upper_singleton(self, v: int):
        # complement of upperset({00 111 00})
        # is lowerset({11 011 11, 11 101 11, 11 110 11})
        # intersect with their union
        inds = support_int(v)
        for i in range(self.n):
            if v & (1 << i):
                inds.append(i)

        prevset = self.set
        self.set = set()
        to_add = set()
        for u in prevset:
            if u & v != v:
                # no intersection, keep
                self.set.add(u)
            else:
                # intersection, split
                for i in inds:
                    to_add.add(u ^ (1 << i))

        for u in to_add:
            if u not in self:
                self.set.add(u)

    def add_lower_singleton(self, v: int, check_new=True):
        if check_new and v in self:
            return
        self.set = {u for u in self.set if not (u & v == u)}
        self.set.add(v)

    def __contains__(self, v: int):
        if v in self.set:
            return True
        for u in self.set:
            if v & u == v:  # v <= u
                return True
        return False


class DynamicUpperSet(DynamicExtremeSet):
    def remove_lower_singleton(self, v):
        pass

    def add_upper_singleton(self, v):
        pass


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


def support_int(v):
    res = []
    for i in range(v):
        b = (1 << i)
        if v & b:
            res.append(i)
        elif v < b:
            break
    return res

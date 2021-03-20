from itertools import combinations
from collections import Counter
from queue import PriorityQueue

from binteger import Bin

from divprop.subsets import (
    DenseSet,
    DivCore_StrongComposition,
    DivCore_StrongComposition8,
    DivCore_StrongComposition16,
    DivCore_StrongComposition32,
    DivCore_StrongComposition64,
    Sbox,
    Sbox32,
    GrowingUpperFrozen,
)

import divprop.logs as logging


def mask(m):
    return (1 << m) - 1


class DivCore:
    """
    Division Core of the S-Box = reduced DPPT of its graph (as a dense set).

    Notation: vectors (u, v), with bit-length (n, m).
    """
    log = logging.getLogger(f"{__name__}:DivCore")

    def __init__(self, data, n, m):
        self._set = set(map(int, data))
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    def to_dense(self):
        # list because swig does not map set straightforwardly.. need a typemap
        return DenseSet(list(self._set), self.n + self.m)

    def to_Bins(self):
        return {Bin(v, self.n+self.m) for v in self._set}

    def __iter__(self):
        return iter(self._set)

    @classmethod
    def from_sbox(cls, sbox: Sbox, n: int = None, m: int = None,
                  method="dense", debug=False):
        if n is None or m is None:
            assert isinstance(sbox, Sbox.classes)
            n = sbox.n
            m = sbox.m
        if not isinstance(sbox, Sbox.classes):
            sbox = Sbox(sbox, n, m)
        method = getattr(cls, "from_sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_sbox:{method}")
        return method(sbox, n, m, debug)

    @classmethod
    def from_sbox_dense(cls, sbox: Sbox, n: int, m: int, debug=False):
        n = int(n)
        m = int(m)

        graph = sbox.graph_dense()

        if debug:
            cls.log.info(f"  graph {graph}")

        graph.do_Mobius()

        if debug:
            cls.log.info(f"    anf {graph}")

        graph.do_MaxSet()

        if debug:
            cls.log.info(f"anf-max {graph}")

        graph.do_Not()

        if debug:
            cls.log.info(f"divcore {graph}")

        return cls(graph, n, m)

    @classmethod
    def from_sbox_peekanfs(cls, sbox: Sbox, n: int, m: int, debug=False):
        n = int(n)
        m = int(m)
        assert n == m, "only bijections supported yet"
        if not isinstance(sbox, Sbox.classes):
            sbox = Sbox(sbox, n, m)
        divcore = SboxPeekANFs(sbox).compute(debug=debug)
        return cls(divcore, n=n, m=m)

    def get_Invalid(self) -> DenseSet:
        """Set I_S from the paper"""
        ret = self.to_dense()
        ret.do_ComplementU2L()
        return ret

    def get_Minimal(self) -> DenseSet:
        """Set M_S from the paper. = MinDPPT up to negating (u)."""
        ret = self.to_dense()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        return ret

    def get_Minimal_Bounds(self) -> (DenseSet, DenseSet):
        """Set M_S from the paper, in the form of its (MinSet,Maxset)"""
        lo = self.to_dense()
        hi = self.get_Minimal()
        hi.do_MaxSet()
        return lo, hi

    def get_Redundant(self) -> DenseSet:
        ret = self.to_dense()
        ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
        ret.do_MinSet()
        return ret

    def get_RedundantAlternative(self) -> DenseSet:
        ret = self.get_Minimal()
        ret.do_MaxSet()
        ret.do_ComplementL2U()
        return ret

    def LB(self) -> DenseSet:
        """
        Outer lower bound for MinDPPT,
        in the form of MaxSet of invalid vectors
        (ones that are a bit lower than vectors from divcore)
        """
        ret = self.to_dense()
        ret.do_ComplementU2L()
        return ret

    def UB(self, method="redundant") -> DenseSet:
        """
        (note: MinDPPT here means extra "not u")

        Outer upper bound for MinDPPT,

        method="redundant"
        in the form of MinSet of redundant vectors
        (ones that are a bit upper in (v) than reduced vectors from divcore)

        method="complement"
        in the form of the complementary MinSet of the MaxSet of MinDPPT

        """
        if method == "redundant":
            ret = self.to_dense()
            ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
            ret.do_MinSet()
        else:
            # ret = self.MinDPPT()
            # ret.do_Not(self.mask_u)
            ret = self.to_dense()
            ret.do_UpperSet(self.mask_u)
            ret.do_MinSet(self.mask_v)

            ret.do_LowerSet()
            ret.do_Complement()
            ret.do_MinSet()
        return ret

    def FullDPPT(self) -> DenseSet:
        """
        DenseSet of all valid transitions, including redundant ones.
        """
        ret = self.to_dense()
        ret.do_UpperSet()
        ret.do_Not(self.mask_u)
        return ret

    def MinDPPT(self) -> DenseSet:
        """
        DenseSet of all valid reduced transitions.
        """
        ret = self.to_dense()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        ret.do_Not(self.mask_u)
        return ret

    def __eq__(self, other):
        assert isinstance(other, DivCore)
        assert self.n == other.n
        assert self.m == other.m
        return self._set == other._set


class SboxPeekANFs:
    log = logging.getLogger(f"{__name__}:SboxPeekANFs")

    def __init__(self, sbox: Sbox, isbox: Sbox = None):
        assert isinstance(sbox, Sbox.classes)
        self.n = int(sbox.n)
        self.sbox = sbox
        self.isbox = ~sbox if isbox is None else isbox

    def compute(self, debug=False):
        n = self.n

        divcore = GrowingUpperFrozen(n=2*n, disable_cache=True)
        divcore.add(frozenset(range(n)))
        divcore.add(frozenset(range(n, 2*n)))

        q = PriorityQueue()
        for i in range(n):
            tocheck = {frozenset((j, n+i)) for j in range(n)}
            q.put((1, False, frozenset({n+i}), tocheck))
            tocheck = {frozenset((n+j, i)) for j in range(n)}
            q.put((1, True, frozenset({i}), tocheck))

        stat = Counter()
        itr = 0
        while q.qsize():
            _, inverse, fset, tocheck = q.get()

            if all(fset2 in divcore for fset2 in tocheck):
                continue

            itr += 1
            self.log.debug(f"run #{itr} fset {fset} inv? {inverse}")
            stat[(inverse, len(fset))] += 1

            mask = Bin(fset, 2*n).int
            if inverse:
                mask >>= n
            res = self.run_mask(mask, inverse=inverse)

            added = {frozenset(Bin(uv, 2*n).support()) for uv in res}
            divcore.update(added)

            if itr % 10 == 0:
                divcore.do_MinSet()

            tocheck -= added
            if not tocheck:
                continue

            rng = range(max(fset)+1, n if inverse else 2*n)
            for i in rng:
                fset2 = fset | {i}
                tocheck2 = {v | {i} for v in tocheck}
                q.put((len(fset)+1, inverse, fset2, tocheck2))

        divcore.do_MinSet()
        statstr = " ".join(
            f"{l}:{cnt}" for (inverse, l), cnt in sorted(stat.items())
        )
        self.log.info(
            f"computed divcore n={n:02d} in {itr} bit-ANF calls, "
            f"stat {statstr}, size {len(divcore)}"
        )
        return set(divcore.to_Bins())

    def get_product(self, mask, inverse):
        sbox = self.isbox if inverse else self.sbox
        return sbox.coordinate_product(mask)

    def run_mask(self, mask, inverse=False):
        assert 0 <= mask < 1 << self.n
        func = self.get_product(mask, inverse)
        func.do_Mobius()
        func.do_MaxSet()
        func.do_Not()
        if inverse:
            return {(mask << self.n) | u for u in func}
        else:
            return {(u << self.n) | mask for u in func}


class HeavyPeeks(SboxPeekANFs):
    log = logging.getLogger(f"{__name__}:HeavyPeeks")

    def __init__(self, n, fws, bks, memorize=False):
        self.n = int(n)
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


def tool_RandomSboxBenchmark():
    import gc
    import sys
    import os
    import subprocess

    n = int(sys.argv[1])

    logging.setup(level="DEBUG")
    logging.addFileHandler(f"divcore_random/{n:02d}")
    log = logging.getLogger(__name__)

    if n >= 24:
        if 0 and not os.path.isfile(f"divcore_random/cache/{n}_i{n-1}.set"):
            filename = f"divcore_random/{n:02d}.sbox"
            ifilename = f"divcore_random/i{n:02d}.sbox"

            log.info("generating...")
            sbox = Sbox32.GEN_random_permutation(n, 2021)
            # sbox = Sbox32.load_from_file(filename)
            log.info(f"generated {n:02d}-bit permutation")

            # sbox.save_to_file(filename)
            log.info(f"saved {n:02d}-bit permutation")

            h = subprocess.check_output(["sha256sum", filename])
            log.info(f"sha256sum: {h}")

            for i in range(n):
                coord = sbox.coordinate(i)
                coord.save_to_file(f"divcore_random/cache/{n}_{i}.set")
                log.info(f"coord {i}/{n} saved")

            log.info("inverting...")
            isbox = sbox
            isbox.invert_in_place()
            log.info("inverting done")
            del sbox
            gc.collect()

            isbox.save_to_file(ifilename)
            log.info(f"saved {n:02d}-bit permutation")

            h = subprocess.check_output(["sha256sum", ifilename])
            log.info(f"sha256sum: {h}")

            for i in range(n):
                coord = isbox.coordinate(i)
                coord.save_to_file(f"divcore_random/cache/{n}_i{i}.set")
                log.info(f"coord {i}/{n} saved")
        else:
            log.info("heavy peeks")
            fws = [f"divcore_random/cache/{n}_{i}.set" for i in range(n)]
            bks = [f"divcore_random/cache/{n}_i{i}.set" for i in range(n)]
            pa = HeavyPeeks(n, fws, bks, memorize=True)
            res = sorted(pa.compute())
            log.info(f"computed divcore: {len(res)} elements")
        quit()

    log.info("generating...")
    sbox = Sbox32.GEN_random_permutation(n, 1)
    log.info(f"generated {n:02d}-bit permutation")

    filename = f"divcore_random/{n:02d}.sbox"
    sbox.save_to_file(filename)
    log.info(f"saved {n:02d}-bit permutation")

    h = subprocess.check_output(["sha256sum", filename])
    log.info(f"sha256sum: {h}")

    pa = SboxPeekANFs(sbox)
    res = sorted(pa.compute())
    log.info(f"computed divcore: {len(res)} elements")

    with open(f"divcore_random/{n:02d}.divcore.sparse", "w") as f:
        print(len(res), file=f)
        for uv in res:
            print(int(uv), file=f, end=" ")

    if n <= 10:
        ans = sorted(DivCore.from_sbox(sbox).to_Bins())
        assert res == ans


if __name__ == '__main__':
    tool_RandomSboxBenchmark()

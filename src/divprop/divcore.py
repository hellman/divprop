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
    Sbox2GraphIndicator,
    Sbox2Coordinates,
    Sbox,
    GrowingUpperFrozen,
    DynamicUpperSet,
)

import divprop.logs as logging


def mask(m):
    return (1 << m) - 1


class DenseDivCore:
    """
    Division Core of the S-Box = reduced DPPT of its graph (as a dense set).

    Notation: vectors (u, v), with bit-length (n, m).

    Wrapper for DenseSet, stored in :attr:`data`.
    """
    log = logging.getLogger(f"{__name__}:DenseDivCore")

    def __init__(self, data, n, m):
        assert isinstance(data, DenseSet)
        assert data.n == n + m
        self.data = data
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    @classmethod
    def from_sbox(cls, sbox, n, m, debug=False, method="simple"):
        method = getattr(cls, "from_sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_sbox:{method}")
        return method(sbox, n, m, debug)

    @classmethod
    def from_sbox_simple(cls, sbox, n, m, debug=False):
        graph = Sbox2GraphIndicator(sbox, n, m)

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

    def get_Invalid(self) -> DenseSet:
        """Set I_S from the paper"""
        return self.data.ComplementU2L()

    def get_Minimal(self) -> DenseSet:
        """Set M_S from the paper. = MinDPPT up to negating (u)."""
        ret = self.data.copy()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        return ret

    def get_Minimal_Bounds(self) -> (DenseSet, DenseSet):
        """Set M_S from the paper, in the form of its (MinSet,Maxset)"""
        ret = self.get_Minimal()
        ret.do_MaxSet()
        return self.data.copy(), ret

    def get_Redundant(self) -> DenseSet:
        ret = self.data.copy()
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
        ret = self.data.copy()
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
            ret = self.data.copy()
            ret.do_UpperSet_Up1(True, self.mask_v)  # is_minset=true
            ret.do_MinSet()
        else:
            # ret = self.MinDPPT()
            # ret.do_Not(self.mask_u)
            ret = self.data.copy()
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
        ret = self.data.copy()
        ret.do_UpperSet()
        ret.do_Not(self.mask_u)
        return ret

    def MinDPPT(self) -> DenseSet:
        """
        DenseSet of all valid reduced transitions.
        """
        ret = self.data.copy()
        ret.do_UpperSet(self.mask_u)
        ret.do_MinSet(self.mask_v)
        ret.do_Not(self.mask_u)
        return ret


class SparseDivCore:
    log = logging.getLogger(f"{__name__}:SparseDivCore")

    def __init__(self, data, n, m):
        self.data = set(data)
        self.n = int(n)
        self.m = int(m)
        self.mask_u = mask(n) << m
        self.mask_v = mask(m)

    @classmethod
    def from_sbox(cls, sbox, n, m, debug=False, method="peekanfs"):
        method = getattr(cls, "from_sbox_" + method)
        if not method:
            raise ValueError(f"Unknown method DenseDivCore.from_sbox:{method}")
        return method(sbox, n, m, debug)

    @classmethod
    def from_sbox_peekanfs(cls, sbox: Sbox, n, m, debug=False):
        assert n == m, "only bijections supported yet"
        divcore = SboxPeekANFs(sbox, n).compute(debug=debug)
        return SparseDivCore(divcore, n=cls.n, m=cls.m)


class SboxPeekANFs:
    log = logging.getLogger(f"{__name__}:SboxPeekANFs")

    def __init__(self, sbox: Sbox):
        self.n = int(sbox.n)
        self.sbox = sbox
        self.isbox = ~sbox

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
            f"computed divcore n={n} in {itr} bit-ANF calls, "
            f"stat {statstr}, size {len(divcore)}"
        )
        return set(divcore.to_Bins())

    def run_mask(self, mask, inverse=False):
        assert 0 <= mask < 1 << self.n
        sbox = self.isbox if inverse else self.sbox
        func = sbox.coordinate_product(mask)
        func.do_Mobius()
        func.do_MaxSet()
        func.do_Not()
        if inverse:
            return {(mask << self.n) | u for u in func}
        else:
            return {(u << self.n) | mask for u in func}


if __name__ == '__main__':
    logging.setup(level="DEBUG")
    if 1:  # present
        n = m = 4
        sbox = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]

    if 1:  # AES
        n = m = 8
        sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]

    import sys
    n = m = int(sys.argv[1])
    sbox = list(range(2**n))
    from random import shuffle
    shuffle(sbox)
    # ans = sorted(DenseDivCore.from_sbox(sbox, n, m).data.to_Bins())

    sbox = Sbox(sbox, n, m)
    pa = SboxPeekANFs(sbox)
    res = sorted(pa.compute())
    # print(*res)
    # assert res == ans
    print("OK")

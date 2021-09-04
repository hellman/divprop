from collections import Counter
from queue import PriorityQueue, Queue

from binteger import Bin

from subsets import DenseSet
from subsets.WeightedSet import GrowingUpperFrozen

from divprop import Sbox

import logging


class SboxPeekANFs:
    """
    Advanced algorithm for DivCore computation,
    by computing backward and forward ANFs of products
    """
    log = logging.getLogger(f"{__name__}:SboxPeekANFs")

    def __init__(self, sbox: Sbox, isbox: Sbox = None):
        assert isinstance(sbox, Sbox.classes)
        self.n = int(sbox.n)
        self.sbox = sbox
        self.isbox = ~sbox if isbox is None else isbox

    def compute(self, debug=False):
        n = self.n

        is_maximal_lb = {}

        divcore = GrowingUpperFrozen(n=2*n, disable_cache=True)
        divcore.add(frozenset(range(n)))
        divcore.add(frozenset(range(n, 2*n)))

        for i in range(n):
            fset = frozenset(range(n)) - {i}
            is_maximal_lb[fset] = 1
            fset = frozenset(range(n, 2*n)) - {n+i}
            is_maximal_lb[fset] = 1

        q = PriorityQueue()
        for i in range(n):
            tocheck = {frozenset((j, n+i)) for j in range(n)}
            q.put((1, False, frozenset({n+i}), tocheck))
            for fset in tocheck:
                is_maximal_lb[fset] = 1
            tocheck = {frozenset((n+j, i)) for j in range(n)}
            q.put((1, True, frozenset({i}), tocheck))
            for fset in tocheck:
                is_maximal_lb[fset] = 1

        # stat = Counter()
        itr = 0
        while q.qsize():
            _, inverse, fset, tocheck = q.get()

            if all(fset2 in divcore for fset2 in tocheck):
                continue

            itr += 1
            self.log.debug(
                f"run #{itr} fset {fset} inv? {inverse}, "
                f"queue size {q.qsize()}"
            )
            # stat[(inverse, len(fset))] += 1

            mask = Bin(fset, 2*n).int
            if inverse:
                mask >>= n
            res_dc = self.run_mask(mask, inverse=inverse)

            added = {frozenset(Bin(uv, 2*n).support) for uv in res_dc}
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

        stat = Counter()
        for uv in divcore:
            u, v = Bin(uv, 2*n).split(2)
            stat[u.hw, v.hw] += 1

        statstr = " ".join(
            f"{a},{b}:{cnt}" for (a, b), cnt in sorted(stat.items())
        )
        self.log.info(
            f"computed divcore n={n:02d} in {itr} bit-ANF calls, "
            f"stat {statstr}, size {len(divcore)}"
        )
        lb = None
        lb = set()
        for i in range(n):
            lb.add(frozenset(range(n)) - {i})
            lb.add(frozenset(range(n, 2*n)) - {n+i})

        added = set()
        q = Queue()
        for i in range(n):
            for j in range(n, 2*n):
                fset = frozenset({i, j})
                q.put(fset)
                added.add(fset)

        while q.qsize():
            fset = q.get()
            if divcore.contains(fset):
                continue

            lb.add(fset)
            for i in fset:
                lb.discard(fset - {i})

            for i in range(2*n):
                if i not in fset:
                    fset2 = fset | {i}
                    if fset2 not in added:
                        q.put(fset2)
                        added.add(fset2)

        return set(divcore.to_Bins()), {Bin(v, 2*n) for v in lb}

    def get_product(self, mask, inverse):
        sbox = self.isbox if inverse else self.sbox
        return sbox.coordinate_product(mask)

    def run_mask(self, mask, inverse=False):
        assert 0 <= mask < 1 << self.n
        func = self.get_product(mask, inverse)
        func.do_ParitySet()
        func.do_MinSet()
        if inverse:
            retdc = {(mask << self.n) | u for u in func}
        else:
            retdc = {(u << self.n) | mask for u in func}
        return retdc

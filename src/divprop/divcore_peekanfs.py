from collections import defaultdict, Counter
from queue import PriorityQueue, Queue

from binteger import Bin

from subsets import DenseSet
from subsets.WeightedSet import GrowingUpperFrozen

from divprop import Sbox

import logging


def half_lens(fset, n):
    l = sum(1 for i in fset if i < n)
    r = len(fset) - l
    return l, r


def is_good(n, fset, cnt):
    l, r = half_lens(fset, n)
    need = l if l > 1 else 0
    need += r if r > 1 else 0
    return cnt == need


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
        self.n_queries = 0

    def compute(self, debug=False, exper=True):
        n = self.n

        if exper:
            # initialize set of parity1 vectors (possibly redundant)
            parity1 = set()
            # - add (1...1, 0...0) to parity1
            parity1.add(frozenset(range(n)))
            # - add (0...0, 1...1) to parity1
            parity1.add(frozenset(range(n, 2*n)))

            # initialize divcore by the two known vectors (1,0), (0,1)
            divcore = parity1.copy()

            # initialize neighbor counter
            cnt = defaultdict(int)

            # cache of masks already run
            masks_run = set()

            # this dict tracks maximal vectors below divcore
            # = maximal vectors of I_S
            invalid_max = set()
            # add predecessors of (1...1, 0...0) and (0...0, 1...1) explicitly
            for i in range(n):
                fset = frozenset(range(n)) - {i}
                invalid_max.add(fset)
                fset = frozenset(range(n, 2*n)) - {n+i}
                invalid_max.add(fset)

            # initialize the exploration queue with (e_i, e_j) pairs
            q = PriorityQueue()
            for i in range(n):
                for j in range(n):
                    fset = frozenset((i, n+j))

                    prio = (1, 1)
                    q.put((prio, fset))

                    cnt[fset] = 0
                    invalid_max.add(fset)

            while q.qsize():
                _, fset = q.get()
                if fset in parity1:
                    divcore.add(fset)
                    invalid_max.discard(fset)
                    continue

                w1, w2 = half_lens(fset, n)

                # choose side to evaluate
                inverse = w1 < w2
                mask = Bin(fset, 2*n).int
                if inverse:
                    mask >>= n
                else:
                    mask &= 2**n-1

                # check if already evaluated
                if (mask, inverse) not in masks_run:
                    masks_run.add((mask, inverse))

                    res = self.run_mask(mask, inverse=inverse)
                    parity1.update({
                        frozenset(Bin(uv, 2*n).support)
                        for uv in res
                    })

                    if fset in parity1:
                        # case of parity 1
                        divcore.add(fset)
                        invalid_max.discard(fset)
                        continue

                # case of parity zero
                for i in range(2*n):
                    if i in fset:
                        # mark downwards as not maximal
                        fset2 = fset - {i}
                        invalid_max.discard(fset2)
                        continue

                    fset2 = fset | {i}
                    assert len(fset2) == len(fset) + 1
                    cnt[fset2] += 1

                    w1, w2 = half_lens(fset2, n)
                    need = w1 if w1 > 1 else 0
                    need += w2 if w2 > 1 else 0
                    if cnt[fset2] == need:
                        prio = min(w1, w2), max(w1, w2)
                        q.put((prio, fset2))
                        invalid_max.add(fset2)

            divcore = {Bin(v, 2*n) for v in divcore}
            invalid_max = {Bin(v, 2*n) for v in invalid_max}
            return divcore, invalid_max

        # this dict tracks maximal vectors below divcore
        # = maximal vectors of I_S
        is_maximal_lb = {}

        divcore = GrowingUpperFrozen(n=2*n, disable_cache=True)
        # add (1...1, 0...0) to divcore
        divcore.add(frozenset(range(n)))
        # add (0...0, 1...1) to divcore
        divcore.add(frozenset(range(n, 2*n)))

        # add predecessors of (1...1, 0...0) and (0...0, 1...1) to I_S explicitly
        for i in range(n):
            fset = frozenset(range(n)) - {i}
            is_maximal_lb[fset] = 1
            fset = frozenset(range(n, 2*n)) - {n+i}
            is_maximal_lb[fset] = 1

        # PriorityQueue:
        # weight, is_inverse, fset (one-side vector-exponent),
        #   tocheck (full vectors made of fset + unit vectors on the other side)
        # (recall that the stop rule is when (e_i, u) are all in divcore)
        # (similarly, all (v, e_i) in divcore for the inverse case)
        q = PriorityQueue()
        # create initial unknown vectors
        # = unit one-sided vectors
        # with all unit-vectors in the other-side tocheck, e.g.
        # - output mask(fset): 10...0
        # - tocheck: (10...0, 10...0), (10...0, 010...0), ..., (10...0, 0...01)
        for i in range(n):
            # forwards
            tocheck = {frozenset((j, n+i)) for j in range(n)}
            q.put((1, False, frozenset({n+i}), tocheck))
            # so far, these vectors can be maximal in I_S
            for fset in tocheck:
                is_maximal_lb[fset] = 1

            # backwards
            tocheck = {frozenset((n+j, i)) for j in range(n)}
            q.put((1, True, frozenset({i}), tocheck))
            # so far, these vectors can be maximal in I_S
            for fset in tocheck:
                is_maximal_lb[fset] = 1

        # stat = Counter()
        itr = 0
        while q.qsize():
            _, inverse, fset, tocheck = q.get()

            # all in divcore already?
            # note that upperset(divcore) is checked
            if all(fset2 in divcore for fset2 in tocheck):
                continue

            itr += 1
            self.log.debug(
                f"run #{itr} fset {fset} inv? {inverse}, "
                f"queue size {q.qsize()}"
            )
            # stat[(inverse, len(fset))] += 1

            # evaluate the MinSet(ParitySet) of S^v or (S^{-1})^u
            mask = Bin(fset, 2*n).int
            if inverse:
                mask >>= n
            res_dc = self.run_mask(mask, inverse=inverse)

            # update current divcore
            added = {frozenset(Bin(uv, 2*n).support) for uv in res_dc}
            divcore.update(added)

            # do minset to keep smaller
            # needed since we often check if vectors are in upperset(divcore)
            # and the sparse algorithm depends on the current size
            if itr % 10 == 0:
                divcore.do_MinSet()

            # what else left to check f
            tocheck -= added
            if not tocheck:
                continue

            # neighbours up of the exponent mask
            # only allowing to increase 0s after the last 1
            # to avoid duplicate visits
            # (i.e. each mask will be constructed by its 1s in order)
            # e.g. for 0101101 we'll do 0100000->0101000->0101100->0101101
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
        self.n_queries += 1
        assert 0 <= mask < 1 << self.n
        func = self.get_product(mask, inverse)
        func.do_ParitySet()
        func.do_MinSet()
        if inverse:
            retdc = {(mask << self.n) | u for u in func}
        else:
            retdc = {(u << self.n) | mask for u in func}
        return retdc


if __name__ == '__main__':
    from random import shuffle, seed, randint
    from divprop import SboxDivision
    from time import time
    t2 = 0
    t3 = 0
    itr = 0
    while 1:
        itr += 1
        s = randint(0, 2**32)
        # s = 1911301760
        print("seed", s)
        seed(s)
        n = randint(3, 10)
        # n = 4
        sbox = list(range(2**n))
        shuffle(sbox)

        sbox = Sbox(sbox, n, n)
        test1 = (
            set(SboxDivision(sbox).divcore.to_Bins()),
            set(SboxDivision(sbox).invalid_max.to_Bins()),
        )

        S2 = SboxPeekANFs(sbox)
        t0 = time()
        test2 = S2.compute(exper=False)
        t2 += time() - t0

        S3 = SboxPeekANFs(sbox)
        t0 = time()
        test3 = S3.compute(exper=True)
        t3 += time() - t0

        print("seed", s)
        assert test1[0] == test2[0]
        assert test1[0] == test3[0]
        assert test1[0] == test2[0] == test3[0]

        assert test1[1] == test2[1]
        assert test1[1] == test3[1]
        assert test1[1] == test2[1] == test3[1]
        print("OK", itr)
        print(S2.n_queries, t2)
        print(S3.n_queries, t3)

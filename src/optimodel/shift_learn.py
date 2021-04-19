import os
from collections import defaultdict
from functools import reduce
import multiprocessing

from binteger import Bin

from subsets import DenseSet, SparseSet
from subsets.learn import Modules as LearnModules

from optimodel.pool import InequalitiesPool, TypeGood, shift_ineq

from divprop.logs import logging

# multiprocessing is nuts
HACK = None


def worker(shift):
    return HACK.worker(shift)


class ShiftLearn:
    log = logging.getLogger(__name__)

    def __init__(self, pool, path, learn_chain):
        self.pool = pool
        if self.pool.is_monotone or self.pool.shift is not None:
            # convert to generic? tool
            raise ValueError(
                "ShiftLearn is only applicable to generic non-shifted sets"
            )

        self.good = DenseSet(list(map(int, self.pool.good)), self.pool.n)
        self.bad = DenseSet(list(map(int, self.pool.bad)), self.pool.n)
        if self.good.Complement() != self.bad:
            self.log.error(
                "the implementation for don't care points"
                "was not carefully checked and tested"
            )

        self.path = path
        self.learn_chain = learn_chain
        assert os.path.isdir(self.path)

    def process_all_shifts(self, threads=1):
        if self.pool.system.is_complete:
            self.log.warning("system is complete, nothign to learn...")
            return

        self.counts = defaultdict(int)
        self.core = {}  # sanity check
        self.solutions = {}

        if threads == 1:
            for shift in self.bad.to_Bins():
                self.log.info(f"processing shift {shift.hex}")
                core, solutions = self.process_shift(shift)

                self.log.info(f"merging solutions of shift {shift.hex}")
                for vec in solutions:
                    if vec not in self.core:
                        self.core.setdefault(vec, core[vec])
                    assert self.core[vec] == core[vec]
                    self.counts[vec] += 1
                self.solutions.update(solutions)
        else:
            shifts = list(self.bad.to_Bins())

            global HACK
            HACK = self
            p = multiprocessing.Pool(processes=threads)
            for shift, core, solutions in p.imap_unordered(worker, shifts):
                self.log.info(f"merging solutions of shift {shift.hex}")
                for vec in solutions:
                    if vec not in self.core:
                        self.core.setdefault(vec, core[vec])
                    assert self.core[vec] == core[vec]
                    self.counts[vec] += 1
                self.solutions.update(solutions)

    def compose(self):
        self.log.info("composing")
        for vec, ineq in self.solutions.items():
            if self.counts[vec] == 2**self.core[vec].weight:
                self.pool.system.add_lower(vec, meta=ineq, is_prime=True)
        self.pool.system.save()

    def worker(self, shift: Bin):
        core, solutions = self.process_shift(shift)
        return shift, core, solutions

    def process_shift(self, shift: Bin):
        subpool = self.process_shift_get_subpool(shift)
        self.log.info(f"extracting solutions for shift {shift.hex}")
        core, solutions = self.extract_subpool_solutions(subpool)
        return core, solutions

    def process_shift_get_subpool(self, shift: Bin):
        # xor
        assert shift.n == self.pool.n
        s = self.good.copy()
        s.do_Not(shift.int)
        s.do_UpperSet()
        good = s.MinSet()
        s.do_Complement()
        removable = s

        bad = self.bad.copy()
        bad.do_Not(shift.int)
        bad &= removable
        # bad.do_LowerSet()  # unnecessary?! optimization

        # good is MinSet of the upper closure
        # bad is what can be removed within this shift
        #          (subset of the removable lower set)

        self.log.info(f"shift {shift.hex} good (MinSet)        {good}")
        self.log.info(f"shift {shift.hex} removable (LowerSet) {removable}")
        self.log.info(f"shift {shift.hex} bad (&LowerSet)      {bad}")

        subpool = InequalitiesPool(
            points_good=good.to_Bins(),
            points_bad=bad.to_Bins(),
            type_good=TypeGood.UPPER,
            sysfile=os.path.join(self.path, f"shift_{shift.hex}.system"),
            pre_shift=shift,
        )

        self.learn_shift(subpool)
        return subpool

    def learn_shift(self, subpool):
        for module, args, kwargs in self.learn_chain:
            if module not in LearnModules:
                raise KeyError(f"Learn module {module} is not registered")
            self.module = LearnModules[module](*args, **kwargs)
            self.module.init(system=subpool.system)
            self.module.learn()

    def extract_subpool_solutions(self, subpool):
        solutions = {}
        core = {}
        for vec in subpool.system.iter_lower():
            qsi = [subpool.i2bad[i].int for i in vec]
            d = DenseSet(qsi, self.pool.n)
            assert d == d.LowerSet(), "temporary assert for no don't care case"
            dmax = d.MaxSet().to_Bins()
            dand = reduce(lambda a, b: a & b, dmax)

            ineq = subpool.system.meta[vec]
            ineq = shift_ineq(ineq, subpool.shift)

            qs = [subpool.i2bad[i] ^ subpool.shift for i in vec]
            mainvec = SparseSet(self.pool.bad2i[q] for q in qs)

            core[mainvec] = dand
            solutions[mainvec] = ineq
        return core, solutions

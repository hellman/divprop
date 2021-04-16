from random import shuffle

from subsets import SparseSet

from divprop.logs import logging

from optisolveapi.milp import MILP
from optisolveapi.sat import CNF

from .utils import truncstr


class LearnModule:
    log = logging.getLogger(f"{__name__}")

    use_point_prec = True

    def init(self, system):
        self._options = self.__dict__.copy()

        self.N = system.n
        self.system = system
        self.oracle = self.system.oracle

        self.milp = None
        self.sat = None

        self.itr = 0
        self.n_upper = 0
        self.n_lower = 0

        self.vec_full = SparseSet(range(self.N))
        self.vec_empty = SparseSet(())

        if self.system.extra_prec is None:
            self.use_point_prec = False

    def learn(self, safe=True):
        self.log.info("===============")
        self.log.info(f"# {type(self).__name__}")
        self.log.info("===============")
        self.log.info(f"options {self._options}")
        self.log.info("starting, stat:")
        self.system.log_info()
        self.log.info("---------------")

        if safe:
            try:
                ret = self._learn()
            except BaseException as error:
                self.log.error(f"learning error {error}, saving")
                self.system.save()
                raise
        else:
            ret = self._learn()

        self.log.info("---------------")
        self.log.info("finished, stat:")
        self.system.save()
        self.log.info("===============")
        self.log.info("")
        return ret

    def query(self, vec):
        if self.use_point_prec:
            vec = self.system.extra_prec.reduce(vec)
        return self.oracle(vec)

    def milp_init(self, maximization=True, init=True):
        if maximization:
            self.milp = MILP.maximization(solver=self.solver)
        else:
            self.milp = MILP.minimization(solver=self.solver)

        self.xs = [self.milp.var_binary("x%d" % i) for i in range(self.N)]
        self.xsum = self.milp.var_int("xsum", lb=2, ub=self.N)
        self.milp.add_constraint(sum(self.xs) == self.xsum)

        if maximization is not None:
            self.milp.set_objective(self.xsum)

        if init:
            self.log.info(
                "milp: initializing "
                f"feasible constraints: {self.system.n_lower()}"
            )
            for vec in self.system.iter_upper():
                self.model_exclude_super(vec)

            self.log.info(
                "milp: initializing "
                f"infeasible constraints: {self.system.n_upper()}"
            )
            for vec in self.system.iter_lower():
                self.model_exclude_sub(vec)

            self.log.info("milp: initialization done")

    def sat_init(self, init_sum=True, init=True):
        self.sat = CNF(solver=self.solver)

        self.xs = [self.sat.var() for i in range(self.N)]
        if init_sum:
            self.xsum = self.sat.SeqAddMany(*[[x] for x in self.xs])
            self.xsum.append(self.sat.ZERO)  # padding

        if init:
            self.log.info(
                "sat: initializing "
                f"feasible constraints: {self.system.n_lower()}"
            )
            for vec in self.system.iter_upper():
                self.model_exclude_super(vec)

            self.log.info(
                "sat: initializing "
                f"infeasible constraints: {self.system.n_upper()}"
            )
            for vec in self.system.iter_lower():
                self.model_exclude_sub(vec)

            self.log.info("sat: initialization done")

    def model_exclude_sub(self, vec):
        if self.use_point_prec:
            vec = self.system.extra_prec.expand(vec)

        if self.milp:
            self.milp.add_constraint(
                sum(self.xs[i] for i in range(self.N) if i not in vec) >= 1
            )

        if self.sat:
            self.sat.add_clause(tuple(
                self.xs[i] for i in self.vec_full - vec
            ))

    def model_exclude_super(self, vec):
        if self.use_point_prec:
            vec = self.system.extra_prec.reduce(vec)

        if self.milp:
            self.milp.add_constraint(
                sum(self.xs[i] for i in vec) <= len(vec) - 1
            )

        if self.sat:
            self.sat.add_clause(tuple(
                -self.xs[i] for i in vec
            ))

    def learn_down(self, vec: SparseSet, meta=None):
        """reduce given upper element to minimal one"""
        if self.system.is_known_upper(vec):
            return

        self.log.debug(
            f"learning down from upper wt {len(vec)}: {truncstr(vec)}"
        )

        inds = list(vec)
        shuffle(inds)
        for i in inds:
            new_vec = vec - i
            assert not self.system.is_known_upper(new_vec)
            if self.system.is_known_lower(new_vec):
                continue

            is_lower, new_meta = self.query(new_vec)
            if is_lower:
                continue

            vec = new_vec
            meta = new_meta

        assert not self.system.is_known_lower(vec)
        assert not self.system.is_known_upper(vec)

        self.system.add_upper(vec, meta=meta, is_prime=True)
        self.model_exclude_super(vec)
        self.log.debug(
            f"learnt minimal upper vec wt {len(vec)}: {truncstr(vec)}"
        )

    def learn_up(self, vec: SparseSet, meta=None):
        """lift given lower element to a maximal one"""
        if self.system.is_known_lower(vec):
            return

        self.log.debug(
            f"learning up from lower wt {len(vec)}: {truncstr(vec)}"
        )

        inds = list(self.vec_full - vec)
        shuffle(inds)
        for i in inds:
            new_vec = vec | i
            assert not self.system.is_known_lower(new_vec)
            if self.system.is_known_upper(new_vec):
                continue

            is_lower, new_meta = self.query(new_vec)
            if not is_lower:
                continue

            vec = new_vec
            meta = new_meta

        assert not self.system.is_known_lower(vec)
        assert not self.system.is_known_upper(vec)

        self.system.add_lower(vec, meta=meta, is_prime=True)
        self.model_exclude_sub(vec)
        self.log.debug(
            f"learnt maximal lower vec wt {len(vec)}: {truncstr(vec)}"
        )

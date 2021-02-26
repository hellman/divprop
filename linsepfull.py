import sage.all
from sage.numerical.mip import MixedIntegerLinearProgram
from sage.numerical.mip import MIPSolverException

from binteger import Bin

from divprop.divcore import DenseDivCore
from divprop.inequalities import LinearSeparator, InequalitiesPool, satisfy
from divprop.sbox_ineqs import get_sbox

from queue import Queue


def neibs_up(v, n):
    for i in range(n):
        if v & (1 << i) == 0:
            yield v ^ (1 << i)


def neibs_down(v, n):
    for i in range(n):
        if v & (1 << i) == 1:
            yield v ^ (1 << i)


class LinearSeparatorFull(LinearSeparator):
    def generate(self):
        n = None
        N = len(self.lo)

        good = {(1<<i): None for i in range(N)}
        good[0] = None

        q = Queue()
        for i in range(N):
            q.put(1<<i)

        n_checks = 0
        checked = set()
        while q.qsize():
            v = q.get()

            is_ok = True
            for lo in neibs_down(v, N):
                if lo not in good:
                    is_ok = False
                    break
            if not is_ok:
                continue

            grp = Bin(v, N).tuple
            assert grp not in checked
            res = self.check_group(grp)
            checked.add(grp)

            n_checks += 1
            if n_checks % 1000 == 0:
                print("stat", n_checks, "checks", len(good), "good", q.qsize(), "queue")
            if not res:
                continue

            good[v] = res

            for j in range(N):
                if (1 << j) > v:
                    up = v | (1 << j)
                    q.put(up)

        print("did", n_checks, "checks")
        tops = set()
        for v, sol in good.items():
            if not any(up in good for up in neibs_up(v, N)):
                # tops[Bin(v, N).tuple] = sol
                print("top", Bin(v, N).str, "%3d" % Bin(v, N).hw(), sol)
                tops.add(sol)
        return tops

    def check_group(self, bads):
        LP = self.model.__copy__()

        assert len(bads) == len(self.lo)
        lstq = [q for take, q in zip(bads, self.lo) if take]
        for i, q in enumerate(lstq):
            LP.add_constraint(self.cs_per_lo[q])

        try:
            LP.solve()
        except MIPSolverException:
            return False

        val_xs = tuple(LP.get_values(x) for x in self.xs)
        val_c = LP.get_values(self.c)
        ineq = val_xs + (-val_c,)
        # ineq = [v*4 for v in ineq]
        # for v in ineq:
        #     # dunno why this hold, the vars are real
        #     assert abs(v - round(v)) < 0.01, ineq
        # ineq = tuple(int(0.5 + v) for v in ineq)
        return ineq


sbox, n, m = get_sbox("aes")
dc = DenseDivCore.from_sbox(sbox, n, m)
mid = dc.MinDPPT().Not(dc.mask_u)
lb = dc.LB()
dclo = mid.MinSet()
dcup = mid.MaxSet()

ret = {}
for typ in "lb", "ubc", "ubo":
    print("\n\n\n")
    print(f"Starting type {typ}")

    if typ == "lb":
        points_good = dc.data
        points_bad = lb

        type_good = "upper"
    elif typ == "ubo":
        points_good = dcup.LowerSet()
        points_bad = points_good.Complement() - lb.LowerSet()

        points_good = points_good.MaxSet()
        points_bad = points_bad.MinSet()

        type_good = "lower"
    elif typ == "ubc":
        points_good = dcup.LowerSet()
        points_bad = points_good.Complement()

        points_good = points_good.MaxSet()
        points_bad = points_bad.MinSet()

        type_good = "lower"
    else:
        assert 0

    print(f"points_good {points_good}")
    print(f"points_bad {points_bad}")
    print(f"inter {points_bad & points_good}")
    assert not (points_bad & points_good)

    points_good = {Bin(p, n+m).tuple for p in points_good}
    points_bad = {Bin(p, n+m).tuple for p in points_bad}

    solver = "glpk"
    if type_good == "lower":
        gen = linsep = LinearSeparatorFull(
            lo=points_good,
            hi=points_bad,
            inverted=True,
            solver=solver,
        )
    elif type_good == "upper":
        gen = linsep = LinearSeparatorFull(
            lo=points_bad,
            hi=points_good,
            inverted=False,
            solver=solver,
        )

    ineqs = L = gen.generate()
    print("got", len(L))
    for p in points_good:
        for ineq in ineqs:
            if not satisfy(p, ineq):
                print("good point removed", p)
                # assert False, (p, ineq)
                # break

    for p in points_bad:
        for ineq in ineqs:
            if not satisfy(p, ineq):
                break
        else:
            print("bad point kept", p)
            # assert False, p
            # break
    break

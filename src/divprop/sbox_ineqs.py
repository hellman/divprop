import argparse
import logging

from binteger import Bin

from divprop.divcore import DenseDivCore
from divprop.inequalities import InequalitiesPool

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# sage/pure python compatibility
try:
    import sage.all
    from sage.crypto.sboxes import sboxes
    is_sage = True
except ImportError:
    logging.warning("running outside of SageMath")
    is_sage = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sbox", type=str,
        help="S-Box (name or comma repr e.g. 2,1,0,3)",
    )
    parser.add_argument(
        "generators", type=str, nargs="?",
        help="Generators with options (available: linsep, random, polyhedron)",
    )
    parser.add_argument(
        "--subset", type=str, default="milp",
        help="Subset algorithm ('milp' or 'greedy:n' where n is number of iter.)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="S-Box (name or comma repr e.g. 2,1,0,3)",
    )

    args = parser.parse_args()

    log.info(args)

    if "," in args.sbox:
        sbox = tuple(map(int, args.sbox.split(",")))
    else:
        sbox = get_sbox(args.sbox)

    if not args.generators:
        generators = DEFAULT_GENS
    else:
        generators = args.generators

    log.info(f"generators: {' '.join(generators)}")

    ret = process_sbox(
        sbox=sbox,
        gens=generators,
        subset_method=args.subset,
        output=args.output,
    )

    print("\n\n")
    print("Inequalities groups:")
    for name, ineqs in ret.items():
        print(name, "= (")
        for ineq in ineqs:
            print(ineq, end=",\n")
        print(")")


def parse_method(s):
    if ":" in s:
        s, str_opts = s.split(":", 1)
        kwargs = {}
        args = []
        for opt in str_opts.split(","):
            if "=" in opt:
                key, val = opt.split("=")
                try:
                    val = int(val)
                except ValueError:
                    pass
                kwargs[key] = val
            else:
                val = opt
                try:
                    val = int(val)
                except ValueError:
                    pass
                args.append(val)
    else:
        args = []
        kwargs = {}
    return s, args, kwargs


DEFAULT_GENS = (
    "polyhedron",
    "linsep:num=250,by_covered=1",
    "linsep:num=50,by_maxsize=1",
    "random:num=10000,max_coef=100,take_best_num=2500",
)

DEFAULT_GENS_LARGE = (
    "linsep:num=250,by_covered=1",
    "linsep:num=50,by_maxsize=1",
    "random:num=10000,max_coef=100,take_best_num=2500",
)

LARGE = 12


def get_sbox(name):
    for k, v in sboxes.items():
        if k.lower() == name.lower():
            return v
    raise KeyError()


def process_sbox(sbox, gens=DEFAULT_GENS, subset_method="milp", output=None):
    sbox = tuple(map(int, sbox))
    n = int(len(sbox)-1).bit_length()
    m = max(int(y).bit_length() for y in sbox)
    assert len(sbox) == 2**n
    assert 0 <= 2**(m-1) <= max(sbox) < 2**m

    if gens is DEFAULT_GENS and n + m >= LARGE:
        gens = DEFAULT_GENS_LARGE

    dc = DenseDivCore.from_sbox(sbox, n, m)
    mid = dc.MinDPPT().Not(dc.mask_u)
    lb = dc.LB()
    dclo = mid.MinSet()
    dcup = mid.MaxSet()

    ret = {}
    for typ in "lb", "ubc", "ubo":
        log.info("\n\n\n")
        log.info(f"Starting type {typ}")

        if typ == "lb":
            points_good = dclo.UpperSet()
            points_bad = points_good.Complement()

            points_good = points_good.MinSet()
            points_bad = points_bad.MaxSet()
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

        log.info(f"points_good {points_good}")
        log.info(f"points_bad {points_bad}")
        log.info(f"inter {points_bad & points_good}")
        assert not (points_bad & points_good)

        points_good = {Bin(p, n+m).tuple for p in points_good}
        points_bad = {Bin(p, n+m).tuple for p in points_bad}

        ret[typ] = Lstar = separate_monotonic(
            points_good=points_good,
            points_bad=points_bad,
            type_good=type_good,
            gens=gens,
            subset_method=subset_method,
        )

        if output:
            filename = f"{output}.{typ}.%d_ineq" % len(Lstar)
            with open(filename, "w") as f:
                print(len(Lstar), file=f)
                for eq in Lstar:
                    print(*eq, file=f)
                    if len(Lstar) < 50:
                        print(eq, end=",\n")
    return ret


def separate_monotonic(
        points_good, points_bad, type_good, gens, subset_method="milp"
    ):
    assert type_good in ("lower", "upper", None)
    subset_method, subset_args, subset_kwargs = parse_method(subset_method)
    assert subset_method in ("milp", "greedy")

    pool = InequalitiesPool(
        points_good=points_good,
        points_bad=points_bad,
        type_good=type_good,
    )

    GENERATORS = {
        "linsep": pool.generate_linsep,
        "random": pool.generate_random,
        "polyhedron": pool.generate_from_polyhedron,
    }
    for gen in gens:
        gen_name, gen_args, gen_kwargs = parse_method(gen)
        GENERATORS[gen_name](*gen_args, **gen_kwargs)

    pool.log_stat()

    pool.check()

    METHODS = {
        "greedy": pool.choose_subset_greedy,
        "milp": pool.choose_subset_milp,
    }
    Lstar = METHODS[subset_method](*subset_args, **subset_kwargs)

    pool.log_stat(Lstar)

    return Lstar

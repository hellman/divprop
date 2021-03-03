import os
import hashlib
from datetime import datetime

import argparse
from argparse import RawTextHelpFormatter

from binteger import Bin

from divprop.divcore import DenseDivCore
from divprop.inequalities import InequalitiesPool, satisfy
import divprop.logging as logging

logging.setup(level="INFO")
log = logging.getLogger(__name__)

# sage/pure python compatibility
try:
    import sage.all
    from sage.crypto.sboxes import sboxes
    is_sage = True
except ImportError:
    logging.warning("running outside of SageMath")
    is_sage = False


DEFAULT_GENS_SMALL = (
    "GemCut",
)

DEFAULT_GENS_MEDIUM = (
    "polyhedron",
    "RandomGroupCut:num=1000,solver=GLPK",
    "RandomPlaneCut:num=10000,max_coef=100,take_best_num=2500",
)

DEFAULT_GENS_LARGE = (
    # "polyhedron",
    "RandomGroupCut:num=1000,solver=GLPK",
    "RandomPlaneCut:num=10000,max_coef=100,take_best_num=2500",
)

MEDIUM = 9
LARGE = 16


DEFAULT_SUBSET = "milp:solver=GLPK"


def main():
    default_chain_small_str = " ".join(DEFAULT_GENS_SMALL)
    default_chain_medium_str = " ".join(DEFAULT_GENS_MEDIUM)
    default_chain_large_str = " ".join(DEFAULT_GENS_LARGE)
    parser = argparse.ArgumentParser(description=f"""
Generate inequalities to model s-box division property propagation.
Default chain:
    {default_chain_small_str}
Default chain for (n+m) >= {MEDIUM}:
    {default_chain_medium_str}
Default chain for (n+m) >= {LARGE}:
    {default_chain_large_str}
Default subset method:
    {DEFAULT_SUBSET}
    """.strip(), formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "sbox", type=str,
        help="S-Box (name or comma repr e.g. 2,1,0,3)",
    )
    parser.add_argument(
        "generators", type=str, nargs="*",
        help="Generators with options (available: linsep, random, polyhedron)",
    )
    parser.add_argument(
        "--subset", type=str, default="milp",
        help="Subset algorithm ('milp' or 'greedy:n' where n is number of iter.)",
    )
    parser.add_argument(
        "--convex", type=str, default="milp",
        help="Subset algorithm ('milp' or 'greedy:n' where n is number of iter.)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="S-Box (name or comma repr e.g. 2,1,0,3)",
    )

    args = parser.parse_args()

    if "," in args.sbox:
        sbox = tuple(map(int, args.sbox.split(",")))
        name = "unknown%x" % hashlib.sha256(str(sbox).encode()).hexdigest()[:8]
    else:
        name = args.sbox.lower()
        sbox = get_sbox(args.sbox)

    if os.path.exists("logs/.divprop"):
        date = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
        logging.addFileHandler(f"logs/{name}.{date}")

    log.info(args)

    n, m = get_sbox_sizes(sbox)

    if not args.generators:
        if n + m < MEDIUM:
            generators = DEFAULT_GENS_SMALL
        elif n + m < LARGE:
            generators = DEFAULT_GENS_MEDIUM
        else:
            generators = DEFAULT_GENS_LARGE

    else:
        generators = args.generators

    log.info(f"generators: {' '.join(generators)}")

    output = args.output
    if output is None and os.path.isfile("results/.divprop"):
        output = f"results/{name}"
        log.info(f"using auto output {output}")

    log.info(f"using output file {output}")

    ret = process_sbox(
        name=args.sbox.lower(),
        sbox=(sbox, n, m),
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


def get_sbox(name):
    for k, sbox in sboxes.items():
        if k.lower() == name.lower():
            sbox = tuple(map(int, sbox))
            return sbox
    raise KeyError()


def get_sbox_sizes(sbox):
    n = int(len(sbox)-1).bit_length()
    m = max(int(y).bit_length() for y in sbox)
    assert len(sbox) == 2**n
    assert 0 <= 2**(m-1) <= max(sbox) < 2**m
    return n, m


def process_sbox(name, sbox, gens=DEFAULT_GENS_LARGE, subset_method="milp", output=None):
    sbox, n, m = sbox
    if output == ".":
        output = f"results/{name}_sbox"

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

    if output:
        if len(ret["ubc"]) < len(ret["ubo"]):
            Lstar = ret["lb"] + ret["ubc"]
        else:
            Lstar = ret["lb"] + ret["ubo"]

        filename = f"{output}.full.%d_ineq" % len(Lstar)
        with open(filename, "w") as f:
            print(len(Lstar), file=f)
            for eq in Lstar:
                print(*eq, file=f)
                if len(Lstar) < 100:
                    print(eq, end=",\n")
    return ret


def separate_monotonic(
        points_good, points_bad, type_good, gens, subset_method=DEFAULT_SUBSET,
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
        "gemcut": pool.generate_GemCut,
        "randomgroupcut": pool.generate_RandomGroupCut,
        "randomplanecut": pool.generate_RandomPlaneCut,
        "polyhedron": pool.generate_from_polyhedron,
    }
    for gen in gens:
        gen_name, gen_args, gen_kwargs = parse_method(gen)
        GENERATORS[gen_name.lower()](*gen_args, **gen_kwargs)

    pool.log_stat()

    pool.check()

    METHODS = {
        "greedy": pool.choose_subset_greedy,
        "milp": pool.choose_subset_milp,
    }
    Lstar = METHODS[subset_method](*subset_args, **subset_kwargs)

    pool.log_stat(Lstar)

    pool.check(Lstar)

    return Lstar

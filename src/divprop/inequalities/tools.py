import os

import argparse
from argparse import RawTextHelpFormatter

from divprop.subsets import DenseSet
from divprop.inequalities import InequalitiesPool
import divprop.logs as logging

log = logging.getLogger(__name__)

# sage/pure python compatibility
try:
    import sage.all
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


def tool_mono2ineqs():
    logging.setup(level="INFO")

    default_chain_small_str = " ".join(DEFAULT_GENS_SMALL)
    default_chain_medium_str = " ".join(DEFAULT_GENS_MEDIUM)
    default_chain_large_str = " ".join(DEFAULT_GENS_LARGE)

    parser = argparse.ArgumentParser(description=f"""
Generate inequalities to model a monotonic set.
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
        "fileprefix", type=str,
        help="Sets prefix "
        "(files with appended .good.set, .bad.set, .type_good must exist)",
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
        "-o", "--output", type=str, default=None,
        help="Output filename",
    )

    args = parser.parse_args()

    assert os.path.exists(args.fileprefix + ".good.set")
    assert os.path.exists(args.fileprefix + ".bad.set")
    assert os.path.exists(args.fileprefix + ".type_good")

    name = args.fileprefix.replace("/", "_")

    if os.path.exists("logs/.divprop"):
        logging.addFileHandler(f"logs/mono2ineqs.{name}")

    log.info(args)

    points_good = DenseSet.load_from_file(args.fileprefix + ".good.set")
    points_bad = DenseSet.load_from_file(args.fileprefix + ".bad.set")

    with open(args.fileprefix + ".type_good") as f:
        type_good = f.read().strip()
        assert type_good in ("upper", "lower")

    log.info(f"points_good: {points_good}")
    if len(points_good) < 100:
        log.info(f"{list(points_good)}")
    log.info(f" points_bad: {points_bad}")
    if len(points_bad) < 100:
        log.info(f"{list(points_bad)}")
    log.info(f"  type_good: {type_good}")

    if type_good == "lower":
        assert points_bad <= points_good.LowerSet().Complement()
    else:
        assert points_bad <= points_good.UpperSet().Complement()

    n = points_good.n
    assert n == points_good.n == points_bad.n

    if not args.generators:
        if n < MEDIUM:
            generators = DEFAULT_GENS_SMALL
        elif n < LARGE:
            generators = DEFAULT_GENS_MEDIUM
        else:
            generators = DEFAULT_GENS_LARGE

    else:
        generators = args.generators

    log.info(f"generators: {' '.join(generators)}")

    output = args.output
    if output is None and os.path.isfile("data/.divprop"):
        output = args.fileprefix + ".ineqs"
        log.info(f"using auto output {output}")

    log.info(f"using output file {output}")

    ineqs = separate_monotonic(
        points_good=points_good.to_Bins(),
        points_bad=points_bad.to_Bins(),
        type_good=type_good,
        gens=generators,
        subset_method=args.subset,
    )

    if output:
        filename = f"{output}.%d" % len(ineqs)
        with open(filename, "w") as f:
            print(len(ineqs), file=f)
            for eq in ineqs:
                print(*eq, file=f)

    log.info("result = (")
    for ineq in ineqs[100:]:
        log.info(f"{ineq},")
    if len(ineqs) > 100:
        log.info("")
    log.info(")")

    print("result = (")
    for ineq in ineqs:
        print(ineq, end=",\n")
    if len(ineqs) > 100:
        print("...,")
    print(")")


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

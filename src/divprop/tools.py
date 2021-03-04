import os
import ast
import hashlib
import argparse

from divprop.all_sboxes import sboxes
from divprop.subsets import DenseSet
from divprop.divcore import DenseDivCore
import divprop.logs as logging


log = logging.getLogger(__name__)


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


def parse_sbox(sbox):
    if "," in sbox:
        sbox = tuple(map(int, ast.literal_eval(sbox)))
        name = "unknown%s" % hashlib.sha256(str(sbox).encode()).hexdigest()[:8]
    else:
        name = sbox.lower()
        sbox = get_sbox(sbox)
    n, m = get_sbox_sizes(sbox)
    return name, sbox, n, m


def tool_sbox2divcore():
    logging.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Generate division core of a given S-box."
    )

    parser.add_argument(
        "sbox", type=str,
        help="S-box (name or python repr e.g. '(2,1,0,3)' )",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: data/divcore.{name}) (.set will be appended)",
    )

    parser.add_argument(
        "-n", "--name", type=str, default=None,
        help="Force name to use in default data paths",
    )

    args = parser.parse_args()

    name, sbox, n, m = parse_sbox(args.sbox)
    try:
        os.mkdir(f"data/{name}")
    except FileExistsError:
        pass

    output = args.output or f"data/{name}/divcore"

    log.info(f"computing division core for '{name}', output to {output}")
    dc = DenseDivCore.from_sbox(sbox, n, m)

    log.info(f"division core: {dc.data}")
    log.info(f"by pairs: {dc.data.str_stat_by_weight_pairs(n, m)}")

    dc.data.save_to_file(output + ".set")
    with open(output + ".dim", "w") as f:
        print(n, m, file=f)


def tool_setinfo():
    logging.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Print information about set (from file)."
    )

    parser.add_argument(
        "filename", type=str, nargs="+",
        help="File with set",
    )
    parser.add_argument(
        "-p", "--print", action="store_true",
        help="Print full set",
    )
    args = parser.parse_args()

    log.info(args)

    for filename in args.filename:
        log.info(f"set file {filename}")
        s = DenseSet.load_from_file(filename)

        log.info(s)

        stat = s.get_counts_by_weights()

        log.info("stat by weights:")
        for u, cnt in enumerate(stat):
            log.info(f"{u} : {cnt}")

        if s.n % 2 == 0:
            n = s.n // 2
            pair_stat = s.get_counts_by_weight_pairs(n, n)
            log.info("stat by pairs:")
            for (u, v), cnt in sorted(pair_stat.items()):
                log.info(f"{u} {v} : {cnt}")

        if args.print:
            print(*s)

        log.info("")


def tool_divcore2bounds():
    logging.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Generate monotone bounds for modeling from division core."
    )

    parser.add_argument(
        "divcore", type=str,
        help="File with division core (.set file, .dim must be present)",
    )

    args = parser.parse_args()

    assert args.divcore.endswith(".set")
    base = args.divcore[:-4]

    divcore_data = DenseSet.load_from_file(args.divcore)
    with open(base + ".dim") as f:
        n, m = map(int, f.read().split())

    log.info(divcore_data)
    log.info(f"n = {n}, m = {m}")

    dc = DenseDivCore(data=divcore_data, n=n, m=m)

    log.info("generating bounds...")

    mid = dc.MinDPPT()
    log.info(f"min-dppt: {mid}")

    mid.do_Not(dc.mask_u)
    lb = dc.LB()
    dclo = dc.data  # = mid.MinSet()
    dcup = mid.MaxSet()

    for typ in "lb", "ubc", "ubo":
        log.info(f"")
        log.info(f"Type {typ}")

        if typ == "lb":
            points_good = dclo
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
        log.info(f"type_good: {type_good}")
        assert not (points_bad & points_good)

        points_good.save_to_file(base + "." + typ + ".good.set")
        points_bad.save_to_file(base + "." + typ + ".bad.set")
        with open(base + "." + typ + ".type_good", "w") as f:
            print(type_good, file=f)


if __name__ == '__main__':
    tool_sbox2divcore()

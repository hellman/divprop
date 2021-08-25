import os
import ast
import hashlib
import argparse

from subsets import DenseSet
from divprop.divprop import Sbox

from divprop.all_sboxes import sboxes
from divprop.divcore import DivCore

import logging
import justlogs

from optimodel.pool import TypeGood

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
    justlogs.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Generate division core of a given S-box."
    )

    parser.add_argument(
        "sbox", type=str,
        help="S-box (name or python repr e.g. '(2,1,0,3)' )",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: data/sbox_{name}/divcore) "
             "(.set will be appended)",
    )

    parser.add_argument(
        "-n", "--name", type=str, default=None,
        help="Force name to use in default data paths",
    )

    args = parser.parse_args()

    name, sbox, n, m = parse_sbox(args.sbox)
    try:
        os.mkdir(f"data/sbox_{name}")
    except FileExistsError:
        pass

    output = args.output or f"data/sbox_{name}/divcore"

    log.info(f"computing division core for '{name}', output to {output}")
    dc = DivCore.from_sbox(Sbox(sbox, n, m))
    data = dc.to_dense()

    log.info(f"division core: {data}")
    log.info(f"by pairs: {data.str_stat_by_weight_pairs(n, m)}")

    data.save_to_file(output + ".set")
    with open(output + ".dim", "w") as f:
        print(n, m, file=f)


def tool_sbox2ddt():
    justlogs.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Generate DDT support of a given S-box."
    )

    parser.add_argument(
        "sbox", type=str,
        help="S-box (name or python repr e.g. '(2,1,0,3)' )",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: data/sbox_{name}/ddt) "
             "(.set will be appended)",
    )

    parser.add_argument(
        "-n", "--name", type=str, default=None,
        help="Force name to use in default data paths",
    )

    args = parser.parse_args()

    name, sbox, n, m = parse_sbox(args.sbox)
    try:
        os.mkdir(f"data/sbox_{name}")
    except FileExistsError:
        pass

    output = args.output or f"data/sbox_{name}/ddt"

    log.info(f"computing ddt support for '{name}', output to {output}")
    ddt = DenseSet(n + m)
    for dx in range(2**n):
        for x in range(2**n):
            dy = sbox[x] ^ sbox[x ^ dx]
            ddt.set((dx << m) | dy)

    log.info(f" ddt: {ddt}")
    log.info(f"~ddt: {ddt.Complement()}")
    log.info(f"by pairs: {ddt.str_stat_by_weight_pairs(n, m)}")

    ddt.save_to_file(output + ".set")
    with open(output + ".dim", "w") as f:
        print(n, m, file=f)

    ddt.save_to_file(output + ".good.set")
    ddt.Complement().save_to_file(output + ".bad.set")
    with open(output + ".type_good", "w") as f:
        print(TypeGood.GENERIC.value, file=f)


# parity transition table?
def tool_sbox2ptt():
    justlogs.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Generate Parity Transition Table (PTT) of a given S-box."
    )

    parser.add_argument(
        "sbox", type=str,
        help="S-box (name or python repr e.g. '(2,1,0,3)' )",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: data/sbox_{name}/ptt) "
             "(.set will be appended)",
    )

    parser.add_argument(
        "-n", "--name", type=str, default=None,
        help="Force name to use in default data paths",
    )

    args = parser.parse_args()

    name, sbox, n, m = parse_sbox(args.sbox)
    try:
        os.mkdir(f"data/sbox_{name}")
    except FileExistsError:
        pass

    output = args.output or f"data/sbox_{name}/ptt"

    log.info(f"computing parity table for '{name}', output to {output}")

    sbox = Sbox(sbox, n, m)
    ptt = DenseSet(n + m)
    # not optimized
    for v in range(2**m):
        comp = sbox.coordinate_product(v)
        comp.do_Mobius()
        for u in comp:
            ptt.set((u << m) | v)

    log.info(f" ptt: {ptt}")
    log.info(f"~ptt: {ptt.Complement()}")
    log.info(f"by pairs: {ptt.str_stat_by_weight_pairs(n, m)}")

    ptt.save_to_file(output + ".set")
    with open(output + ".dim", "w") as f:
        print(n, m, file=f)

    ptt.save_to_file(output + ".good.set")
    ptt.Complement().save_to_file(output + ".bad.set")
    with open(output + ".type_good", "w") as f:
        print(TypeGood.GENERIC.value, file=f)
    del ptt

    # save Minimal too
    output = f"data/sbox_{name}/ndppt"

    dc = DivCore.from_sbox(sbox)
    MS = dc.get_Minimal()
    MS.save_to_file(output + ".set")

    log.info(f" minimal: {MS}")
    log.info(f"~minimal: {MS.Complement()}")
    log.info(f"by pairs: {MS.str_stat_by_weight_pairs(n, m)}")

    MS.save_to_file(output + ".set")
    with open(output + ".dim", "w") as f:
        print(n, m, file=f)

    MS.save_to_file(output + ".good.set")
    MS.Complement().save_to_file(output + ".bad.set")
    with open(output + ".type_good", "w") as f:
        print(TypeGood.GENERIC.value, file=f)


def tool_setinfo():
    justlogs.setup(level="INFO")

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
    parser.add_argument(
        "-s", "--short", action="store_true",
        help="Print one-line description per set",
    )
    args = parser.parse_args()

    log.info(args)
    if args.short:
        DenseSet.set_quiet()

    mxlen = max(map(len, args.filename))
    for filename in args.filename:
        if not args.short:
            log.info(f"set file {filename}")
        s = DenseSet.load_from_file(filename)

        log.info(f"{filename.rjust(mxlen)}: {s}")
        if args.short:
            continue

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
    justlogs.setup(level="INFO")

    parser = argparse.ArgumentParser(
        description="Generate monotone bounds for modeling from division core."
    )

    parser.add_argument(
        "-t", "--type", default="lb,ubc,ubo",
        help="Type of bounds to store, comma-separated: "
             "'lb', 'ubc', 'ubo', 'full'"
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

    dc = DivCore(data=divcore_data, n=n, m=m)

    log.info("generating bounds...")

    mid = dc.MinDPPT()
    log.info(f"min-dppt: {mid}")
    mid.do_Not(dc.mask_u)
    log.info(f"     M_S: {mid}")

    lb = dc.LB()
    assert mid.UpperSet().Complement().MaxSet() == lb
    dclo = divcore_data  # = mid.MinSet()
    dcup = mid.MaxSet()

    inter = (dcup.LowerSet().Complement() & dclo.UpperSet().Complement())
    print("intersection", inter)

    typs = args.type.lower().split(",")
    for typ in typs:
        log.info("")
        log.info(f"Type {typ}")

        if typ == "lb":
            points_good = dclo
            points_bad = lb

            assert points_bad == dc.get_Invalid()

            type_good = TypeGood.UPPER
        elif typ == "ubo":
            points_good = dcup.LowerSet()
            points_bad = points_good.Complement() - lb.LowerSet()

            points_good = points_good.MaxSet()
            points_bad = points_bad.MinSet()

            assert points_bad == dc.get_Redundant()

            type_good = TypeGood.LOWER
        elif typ == "ubc":
            points_good = dcup.LowerSet()
            points_bad = points_good.Complement()

            points_good = points_good.MaxSet()
            points_bad = points_bad.MinSet()

            assert points_bad == dc.get_RedundantAlternative()

            type_good = TypeGood.LOWER
        elif typ == "full":
            points_good = mid
            points_bad = mid.Complement()

            type_good = TypeGood.GENERIC
        else:
            assert 0

        log.info(f"points_good {points_good}")
        log.info(f"points_bad {points_bad}")
        log.info(f"type_good: {type_good}")
        assert not (points_bad & points_good)

        points_good.save_to_file(base + "." + typ + ".good.set")
        points_bad.save_to_file(base + "." + typ + ".bad.set")
        with open(base + "." + typ + ".type_good", "w") as f:
            print(type_good.value, file=f)


if __name__ == '__main__':
    tool_sbox2ddt()

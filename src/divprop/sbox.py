import hashlib

import argparse

from divprop.all_sboxes import sboxes
from divprop.divcore import DenseDivCore
import divprop.logging as logging


logging.setup(level="INFO")
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
        sbox = tuple(map(int, sbox.split(",")))
        name = "unknown%x" % hashlib.sha256(str(sbox).encode()).hexdigest()[:8]
    else:
        name = sbox.lower()
        sbox = get_sbox(sbox)
    n, m = get_sbox_sizes(sbox)
    return name, sbox, n, m


def tool_sbox2divcore():
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

    args = parser.parse_args()

    name, sbox, n, m = parse_sbox(args.sbox)
    output = args.output or f"data/divcore.{name}"

    log.info("computing division core")
    dc = DenseDivCore.from_sbox(sbox, n, m)
    log.info("division core: {dc.data}")
    pair_stat = dc.data.get_counts_by_weight_pairs()
    pair_stat = " ".join("%d,%d:%d" for (u, v), cnt in pair_stat.items())
    log.info("division core: {pair_stat}")
    dc.data.save_to_file(output)


if __name__ == '__main__':
    tool_sbox2divcore()

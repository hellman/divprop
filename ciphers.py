from itertools import product
from divprop.divcore import DivCore_StrongComposition16
import divprop.logging as logging

log = logging.getLogger(__name__)

PRESENT_SBOX = 0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2


class Sandwich:
    def __init__(self, n, r, m, part1, keys, part2):
        self.n = n
        self.r = r
        self.m = m
        self.part1 = tuple(map(int, part1))
        self.keys = tuple(map(int, keys))
        self.part2 = tuple(map(int, part1))
        assert len(self.part1) == 2**n
        assert 0 <= min(self.part1) <= max(self.part1) < 2**r
        assert 0 <= min(self.keys) <= max(self.keys) < 2**r
        assert len(self.part2) == 2**r
        assert 0 <= min(self.part2) <= max(self.part2) < 2**m

    def compute_divcore(self, chunk=128, filename=None):
        DCS = DivCore_StrongComposition16(
            self.n, self.r, self.m,
            self.part1, self.part2,
        )
        DCS.set_keys(self.keys)
        DCS.shuffle()
        log.info(
            f"processing Sandwich({self.n},{self.r},{self.m})"
            f" with {len(self.keys)} keys, saving to {filename}"
        )
        n_done = 0
        while len(DCS.keys_left):
            DCS.process(chunk)
            n_done += chunk
            log.info(f"done {n_done}/{len(self.keys)}: {DCS.divcore}")
            if filename:
                DCS.divcore.save_to_file(filename)


class SSB16:
    sbox = NotImplemented
    poly = None
    n = r = m = 16

    def __init__(self):
        if self.poly:
            self.MULTAB = self._precomp_multab()
        self.SB = self.compute(self.sub)
        self.MC = self.compute(self.mix)
        self.SBMC = self.compute(self.sub, self.mix)
        self.MCSB = self.compute(self.mix, self.sub)

    def get_keys(self):
        return list(range(2**self.r))

    def mix(self, a, b, c, d):
        raise NotImplementedError()

    def sub(self, a, b, c, d):
        return tuple(map(self.sbox.__getitem__, (a, b, c, d)))

    def compute(self, *funcs):
        s = []
        for a, b, c, d in product(range(16), repeat=4):
            for func in funcs:
                a, b, c, d = func(a, b, c, d)
            s.append((a << 12) | (b << 8) | (c << 4) | d)
        return s

    def _precomp_multab(self):
        tab = {}
        for a, b in product(range(4), repeat=2):
            key = a, b
            res = 0
            while a:
                if a & 1:
                    res ^= b
                a >>= 1
                b <<= 1
                if b >> 4:
                    b ^= self.poly
            tab[key] = res
        return tab

    def make_sandwich(self):
        return Sandwich(
            16, 16, 16,
            self.SB,
            self.get_keys(),
            self.SBMC,
        )


class Midori64(SSB16):
    # same as MANTIS, CRAFT
    sbox = 0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6

    def mix(self, a, b, c, d):
        t = a ^ b ^ c ^ d
        a ^= t
        b ^= t
        c ^= t
        d ^= t
        return a, b, c, d


class LED(SSB16):
    sbox = PRESENT_SBOX
    poly = 0x13

    def mix(self, a, b, c, d):
        for i in range(4):
            tmp = (
                self.MULTAB[a, 4] ^ self.MULTAB[b, 1]
                ^ self.MULTAB[c, 2] ^ self.MULTAB[d, 2]
            )
            a, b, c, d = b, c, d, tmp
        return a, b, c, d


class SKINNY64(SSB16):
    sbox = 0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf

    def mix(self, a, b, c, d):
        b ^= c
        c ^= a
        d ^= c
        a, b, c, d = d, a, b, c
        return a, b, c, d

    def get_keys(self):
        return [(a << 8) | 0x20 for a in range(16 * 16)]


ciphers = {
    cls.__name__.lower(): cls
    for cls in (Midori64, LED, SKINNY64)
}

# class Rectangle:
#     sbox = 0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9, 0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2

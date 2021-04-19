from itertools import chain

from binteger import Bin


class SparseSet(tuple):
    def __new__(cls, vec):
        self = super().__new__(cls, sorted(map(int, vec)))
        if self:
            for a in self:
                assert 0 <= a
            # ensure sorted for possible merge-like operations
            # also checks for dups (3, 3 ambiguous - xor or or?)
            for a, b in zip(self, self[1:]):
                assert a < b
        return self

    def as_Bin(self, n):
        return Bin(set(self[:]), n)

    def _coerce(self, other):
        if isinstance(other, (list, set, tuple)):
            return SparseSet(other)
        if isinstance(other, int):
            return SparseSet((other,))
        if not isinstance(other, SparseSet):
            raise TypeError("Can not coerce")
        return other

    def __or__(self, other):
        """
        >>> SparseSet((0, 2, 5)) | 1
        SparseSet((0, 1, 2, 5))
        >>> SparseSet((0, 2, 5)) | 2
        SparseSet((0, 2, 5))
        >>> SparseSet((0, 2, 5)) | (1, 2, 3)
        SparseSet((0, 1, 2, 3, 5))
        >>> SparseSet((0, 2, 5)) | {2, 5, 7, 8}
        SparseSet((0, 2, 5, 7, 8))
        >>> SparseSet((0, 2, 5)) | [5, 1]
        SparseSet((0, 1, 2, 5))
        """
        other = self._coerce(other)
        return SparseSet(set(chain(self, other)))
    __ror__ = __or__

    def __and__(self, other):
        """
        >>> SparseSet((0, 2, 5)) & 1
        SparseSet(())
        >>> SparseSet((0, 2, 5)) & 2
        SparseSet((2,))
        >>> SparseSet((0, 2, 5)) & (1, 2, 3, 5)
        SparseSet((2, 5))
        >>> SparseSet((0, 2, 5)) & (0, 1, 2, 3, 5)
        SparseSet((0, 2, 5))
        >>> SparseSet((0, 2, 5)) & (1, 3, 4, 6, 7)
        SparseSet(())
        """
        other = self._coerce(other)
        return SparseSet(set(self) & set(other))
    __rand__ = __and__

    def __xor__(self, other):
        """
        >>> SparseSet((0, 2, 5)) ^ 1
        SparseSet((0, 1, 2, 5))
        >>> SparseSet((0, 2, 5)) ^ 2
        SparseSet((0, 5))
        >>> SparseSet((0, 2, 5)) ^ (1, 2, 3, 5)
        SparseSet((0, 1, 3))
        >>> SparseSet((0, 2, 5)) ^ (1, 3, 4, 6, 7)
        SparseSet((0, 1, 2, 3, 4, 5, 6, 7))
        """
        other = self._coerce(other)
        return SparseSet(set(self) ^ set(other))
    __rand__ = __and__

    def __sub__(self, other):
        """
        >>> SparseSet((0, 2, 5)) - 2
        SparseSet((0, 5))
        >>> SparseSet((0, 2, 5)) - 3
        SparseSet((0, 2, 5))
        >>> SparseSet((0, 2, 5)) - (3, 2)
        SparseSet((0, 5))
        """
        other = self._coerce(other)
        return SparseSet(set(self) - set(other))

    def __rsub__(self, other):
        other = self._coerce(other)
        return SparseSet(set(other) - set(self))

    def __add__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        """
        >>> SparseSet((0, 1, 5)) <= SparseSet((0, 1, 2, 3, 4, 5))
        True
        >>> SparseSet((0, 1, 5)) <= SparseSet((0, 1, 5))
        True
        >>> SparseSet((0, 1, 5)) <= SparseSet((0, 1))
        False
        >>> SparseSet((0, 1, 5)) <= SparseSet((0, 1, 2, 3, 4))
        False
        >>> SparseSet((0, 1, 5)) <= (0, 1, 2, 3, 4, 5)
        True
        >>> SparseSet((0, 1, 5)) <= (0, 1)
        False
        """
        other = self._coerce(other)
        return set(self) <= set(other)

    def __ge__(self, other):
        other = self._coerce(other)
        return set(self) >= set(other)

    def __lt__(self, other):
        """
        >>> SparseSet((0, 1, 5)) < SparseSet((0, 1, 2, 3, 4, 5))
        True
        >>> SparseSet((0, 1, 5)) < SparseSet((0, 1, 5))
        False
        >>> SparseSet((0, 1, 5)) < SparseSet((0, 1))
        False
        >>> SparseSet((0, 1, 5)) < SparseSet((0, 1, 2, 3, 4))
        False
        >>> SparseSet((0, 1, 5)) < (0, 1, 2, 3, 4, 5)
        True
        >>> SparseSet((0, 1, 5)) < (0, 1, 5)
        False
        """
        other = self._coerce(other)
        return set(self) < set(other)

    def __gt__(self, other):
        other = self._coerce(other)
        return set(self) > set(other)

    def __repr__(self):
        return f"{type(self).__name__}({self[:]})"

    def __str__(self):
        return ",".join(map(str, self))

    def to_Bin(self, n):
        """
        >>> SparseSet((1, 2, 5)).to_Bin(10)
        Bin(0b0110010000, n=10)
        """
        return Bin(set(self), n=n)

    def neibs_down(self):
        """
        >>> sorted(SparseSet((1, 2, 5)).neibs_down(), key=str)
        [SparseSet((1, 2)), SparseSet((1, 5)), SparseSet((2, 5))]
        """
        for v in self:
            yield self - v

    def neibs_up(self, n):
        """
        >>> sorted(SparseSet((1, 2)).neibs_up(4), key=str)
        [SparseSet((0, 1, 2)), SparseSet((1, 2, 3))]
        """
        me = set(self)
        for v in range(n):
            if v not in me:
                yield self | v

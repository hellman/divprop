class WeightedSet:
    def __init__(self, spec, n):
        self.n = int(n)
        self.sets = [set() for i in range(self.n+1)]
        for v in spec:
            self.add(v)

    def add(self, v):
        self.sets[len(v)].add(v)

    def iter_ge(self, w=0):
        for s in self.sets[w:]:
            yield from s

    def iter_le(self, w=None):
        if w is None:
            w = self.n
        for s in self.sets[:w+1]:
            yield from s

    def iter_wt(self, w):
        yield from self.sets[w]

    def __iter__(self):
        return iter(self.iter_ge())

    def __contains__(self, v):
        return v in self.sets[len(v)]

    def remove(self, v):
        s = self.sets[len(v)]
        return s.remove(v)

    def __len__(self):
        return sum(len(v) for v in self.sets)

    def do_MaxSet(self):
        w = max(w for w in range(self.n+1) if self.sets[w])
        # remove from w1 using w2
        for w2 in reversed(range(w+1)):
            for w1 in reversed(range(w2)):
                if not self.sets[w1]:
                    continue
                self.sets[w1] = {
                    v for v in self.sets[w1]
                    if not any(v & u == v for u in self.sets[w2])
                }

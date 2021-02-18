#pragma once

#include "common.hpp"
#include "DenseSet.hpp"

template<typename T>
DenseSet Sbox2GraphIndicator(const std::vector<T> &sbox, int n, int m) {
    ensure(sbox.size() == (1ull << n));

    uint64_t max_y = 1ull << m;

    DenseSet graph(n + m);
    fori (x, 1ull << n) {
        uint64_t y = (uint64_t)sbox[x];
        ensure(y <= max_y);
        graph.set((u64(x) << m) | y);
    }
    return graph;
}
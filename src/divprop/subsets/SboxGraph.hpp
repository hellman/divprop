#pragma once

#include "common.hpp"
#include "DenseSet.hpp"

template<typename T>
DenseSet T_Sbox2GraphIndicator(const std::vector<T> &sbox, int n, int m) {
    ensure(0 <= n && n <= 63);
    ensure(0 <= m && m <= 64);
    ensure(sbox.size() == (1ull << n));
    uint64_t max_y = (m == 64) ? (-1ull) : (1ull << m);

    DenseSet graph(n + m);
    fori (x, 1ull << n) {
        uint64_t y = (uint64_t)sbox[x];
        ensure(y <= max_y);
        graph.set((u64(x) << m) | y);
    }
    return graph;
}

template<typename T>
std::vector<DenseSet> T_Sbox2Coordinates(const std::vector<T> &sbox, int n, int m) {
    ensure(0 <= n && n <= 63);
    ensure(0 <= m && m <= 64);
    ensure(sbox.size() == (1ull << n));
    uint64_t max_y = (m == 64) ? (-1ull) : (1ull << m);

    std::vector<DenseSet> funcs(m, DenseSet(n));
    fori (x, 1ull << n) {
        uint64_t y = (uint64_t)sbox[x];
        ensure(y <= max_y);
        fori (i, m) {
            funcs[m-1-i].set(x, (y >> i) & 1);
        }
    }
    return funcs;
}

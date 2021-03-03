#pragma once

#include "common.hpp"
#include "DenseSet.hpp"

struct DivCore_StrongComposition {
    int n, r, m;
    std::vector<DenseSet> current;
    std::vector<uint64_t> tab1;
    std::vector<uint64_t> tab2;
    std::vector<uint64_t> keys_left;
    DenseSet _ones;

    DenseSet divcore;

    DivCore_StrongComposition(int _n, int _r, int _m, const std::vector<uint64_t> &_tab1, const std::vector<uint64_t> &_tab2);
    void shuffle();
    void process(uint64_t num = -1ull);
    void _process_key(uint64_t key);
    void _finalize();
};
#include <random>
#include <algorithm>

#include "DivCore.hpp"

DivCore_StrongComposition::DivCore_StrongComposition(
    int _n, int _r, int _m,
    const vector<uint64_t> &_tab1, const vector<uint64_t> &_tab2
) :
n(_n), r(_r), m(_m), tab1(_tab1), tab2(_tab2), _ones(n), divcore(n + m)
{
    fori (k, 1ull << r) {
        keys_left.push_back(k);
    }
    _ones.fill();
    current.assign(1ull << m, DenseSet(n));
}

void DivCore_StrongComposition::shuffle() {
    std::shuffle(keys_left.begin(), keys_left.end(), default_random_engine());
}
void DivCore_StrongComposition::process(uint64_t num) {
    num = min(num, keys_left.size());
    vector<uint64_t> keys(keys_left.end() - num, keys_left.end());
    #pragma omp parallel for
    for (auto key: keys) {
        _process_key(key);
    }
    _finalize();
}
void DivCore_StrongComposition::_process_key(uint64_t key) {
    vector<DenseSet> products(m, DenseSet(n));

    // compute single bit products
    // could compute all products in linear time
    // but increases memory usage
    // NOTE: reverse order (LSB to MSB)
    fori (i, m) {
        fori (x, 1ull << n) {
            uint64_t y = tab2[tab1[x] ^ key];
            int bit = (y >> i) & 1;
            products[i].set(x, bit);
        }
    }

    // alternative idea to improve cache usage:
    // precompute products say for current bunch of keys
    // iterate over v
    // inside each v run omp-for over keys
    // every thread works with one local func and one global func same for all threads

    DenseSet cur(n);
    fori (v, 1ull << m) {
        cur = _ones;
        auto tmp = v;
        fori (i, m) {
            if (tmp & 1) {
                cur &= products[i];
            }
            tmp >>= 1;
        }

        cur.do_Mobius();

        // TBD: maybe make maxset here and iterate over support?

        #pragma omp critical(orthings)
        current[v] |= cur;
    }
}
void DivCore_StrongComposition::_finalize() {
    fori (v, 1ull << m) {
        // since we have to iterate over bits anyway
        // better to reduce the size
        current[v].do_MaxSet();
        for (auto u: current[v].get_support()) {
            divcore.set((u << m) | v);
        }
    }
    divcore.do_MinSet();
}

// DenseSet DivCoreLB_chunked(const vector<uint64_t> &ks, const vector<uint64_t> tab1) {
//     uint64_t itr = -1ull;
//     for (auto k: ks) {
//         itr++;

//         fprintf(stderr, "itr %lu k %04lx\n", itr, k);
//         uint16_t tab[1ull << 16];
//         fori (x, 1ull<<n) {
//             tab[x] = cipher::evaluate(x, k);
//         }

//         curs[0].data = ones.data;
//         fori (j, n) {
//             fori (x, 1ull<<n) {
//                 int bit = (tab[x] >> j) & 1;
//                 curs[1ull << j].set(x, bit);
//             }
//             fori (i, 1, 1ull << j) {
//                 fori (t, 1ull << (n - 6)) {
//                     curs[i + (1ull << j)].data[t] = curs[i].data[t] & curs[1ull << j].data[t];
//                 }
//             }
//         }

//         fori (v, 1ull << n) {
//             curs[v].do_Mobius();

//             uint64_t nv = (~v) & ((1ull << n) - 1);
//             auto off = upper_all.data.data() + (nv << (m - 6));
//             fori (i, curs[v].data.size()) {
//                 off[i] |= curs[v].data[i];
//             }

//             // for (auto u: curs[v].get_support()) {
//             //     uint64_t nu = (~u) & ((1ull << n) - 1);
//             //     upper_all.set((nu << m) | v);
//             // }
//         }

//         if (itr && itr % 8 == 0 && 0) {
//             upper_all.do_MaxSet();

//             fprintf(stderr, "itr %lu : %lu elements in divcore\n", itr, upper_all.get_weight());
//         }
//     }
//     // upper_all.do_MaxSet();
//     // upper_all.do_Not();
//     return upper_all;
// }


// DenseSet DivCoreLB_chunked_threaded(vector<uint64_t> ks, int nthreads) {
//     vector<future<DenseSet>> results(nthreads);
//     fori (i, nthreads) {
//         vector<uint64_t> subks;
//         fori (j, i, ks.size(), nthreads)
//             subks.push_back(ks[j]);
//         // results.push_back(async(launch::async, DivCoreLB_direct<cipher>, subks));
//         results.push_back(async(launch::async, DivCoreLB_chunked<cipher>, subks));
//     }

//     DenseSet core = results.back().get();
//     results.pop_back();
//     fori (i, nthreads-1) {
//         core |= results.back().get();
//         results.pop_back();
//     }
//     fprintf(stderr, "DivisionCore merged: %lu\n", core.get_weight());
//     core.do_MaxSet();
//     core.do_Not();
//     fprintf(stderr, "DivisionCore inverted: %lu\n", core.get_weight());
//     core.do_MinSet();
//     fprintf(stderr, "DivisionCore final: %lu\n", core.get_weight());
//     return core;
// }
#include "common.hpp"

#include "Sweep.hpp"

#include "DenseSet.hpp"


DenseSet::DenseSet() {
    n = 0;
}
DenseSet::DenseSet(int _n) {
    ensure(_n >= 0 and _n <= 64, "supported set dimension is between 1 and 64"); // not supported
    n = _n;
    clear();
}
DenseSet DenseSet::copy() const {
    return *this;
}
void DenseSet::free() {
    data.clear();
    n = 0;
}
void DenseSet::clear() {
    data.assign(HICEIL(1ull << n), 0);
}

// ========================================
// Single bit get/set
// ========================================

int DenseSet::get(u64 x) const {
    return (data[HI(x)] >> LO(x)) & 1;
}
void DenseSet::set(u64 x) {
    data[HI(x)] |= 1ull << LO(x);
}
void DenseSet::set(u64 x, u64 value) {
    data[HI(x)] &= ~(1ull << LO(x));
    data[HI(x)] |= (value & 1ull) << LO(x);
}

// ========================================
// Tools
// ========================================
template<auto func>
void DenseSet::do_Sweep(u64 mask) {
    mask &= (1ull << n)-1;
    // we can use GenericSweep
    // pretending we have bit-slice 64 parallel sets in our array
    if (HI(mask)) {
        GenericSweep<func>(data, HI(mask));
    }
    // and then it's only left to Sweep each word
    if (LO(mask)) {
        fori (i, data.size()) {
            GenericSweepWord<func>(data[i], LO(mask));
        }
    }
}
// for python low-level API
void DenseSet::do_Sweep_OR_up(uint64_t mask) {
    do_Sweep<OR_up<u64>>(mask);
}
void DenseSet::do_Sweep_OR_down(uint64_t mask) {
    do_Sweep<OR_down<u64>>(mask);
}
void DenseSet::do_Sweep_XOR_up(uint64_t mask) {
    do_Sweep<XOR_up<u64>>(mask);
}
void DenseSet::do_Sweep_XOR_down(uint64_t mask) {
    do_Sweep<XOR_down<u64>>(mask);
}
void DenseSet::do_Sweep_AND_up(uint64_t mask) {
    do_Sweep<AND_up<u64>>(mask);
}
void DenseSet::do_Sweep_AND_down(uint64_t mask) {
    do_Sweep<AND_down<u64>>(mask);
}
void DenseSet::do_Sweep_SWAP(uint64_t mask) {
    do_Sweep<SWAP<u64>>(mask);
}
void DenseSet::do_Sweep_LESS_up(uint64_t mask) {
    do_Sweep<LESS_up<u64>>(mask);
}
void DenseSet::do_Sweep_MORE_down(uint64_t mask) {
    do_Sweep<MORE_down<u64>>(mask);
}

// ========================================
// Bitwise
// ========================================
bool DenseSet::is_compatible_set(const DenseSet & b) const {
    return n == b.n && data.size() == b.data.size();
}
bool DenseSet::operator==(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        return false;
    }
    return true;
}
bool DenseSet::operator!=(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        return true;
    }
    return false;
}
bool DenseSet::operator<(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        if ((data[i] & data2[i]) == data[i]) return true; // prec
        return false;
    }
    return false;
}
bool DenseSet::operator>(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        if ((data[i] & data2[i]) == data2[i]) return true; // succ
        return false;
    }
    return false;
}
bool DenseSet::operator<=(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        if ((data[i] & data2[i]) == data[i]) return true; // prec
        return false;
    }
    return true;
}
bool DenseSet::operator>=(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        if ((data[i] & data2[i]) == data2[i]) return true; // succ
        return false;
    }
    return true;
}
DenseSet & DenseSet::operator|=(const DenseSet & b) {
    ensure(is_compatible_set(b));
    fori (i, data.size()) {
        data[i] |= b.data[i];
    }
    return *this;
}
DenseSet & DenseSet::operator^=(const DenseSet & b) {
    ensure(is_compatible_set(b));
    fori (i, data.size()) {
        data[i] ^= b.data[i];
    }
    return *this;
}
DenseSet & DenseSet::operator&=(const DenseSet & b) {
    ensure(is_compatible_set(b));
    fori (i, data.size()) {
        data[i] &= b.data[i];
    }
    return *this;
}
DenseSet & DenseSet::operator-=(const DenseSet & b) {
    ensure(is_compatible_set(b));
    fori (i, data.size()) {
        data[i] &= ~b.data[i];
    }
    return *this;
}

DenseSet DenseSet::get_head_fixed(int h, u64 value) {
    ensure(value < (1ull << h));
    ensure(h >=0 && h <= n);
    ensure(n - h >= 6);
    DenseSet result(n - h);
    u64 start = value << (n - h - 6);
    u64 end = (value + 1) << (n - h - 6);
    result.data = vector<u64>(data.begin() + start, data.begin() + end);
    return result;
}

// ========================================
// Support
// ========================================

void DenseSet::iter_support(function<void(u64)> const & func) const {
    fori (hi, data.size()) {
        if (data[hi]) {
            fori (lo, 64) {
                if ((data[hi] >> lo) & 1) {
                    u64 x = (hi << 6) | lo;
                    func(x);
                }
            }
        }
    }
}

// returns support of the function
// 32-bit version useful? if support is small then 2x RAM does not matter
// if it's large then working with DenseSet is better anyway...
vector<u64> DenseSet::get_support() const {
    vector<u64> inds;
    fori (hi, data.size()) {
        if (data[hi]) {
            fori (lo, 64) {
                if ((data[hi] >> lo) & 1) {
                    u64 ind = (hi << 6) | lo;
                    inds.push_back(ind);
                }
            }
        }
    }
    return inds;
}

u64 DenseSet::get_weight() const {
    u64 cnt = 0;
    for (auto word: data) {
        if (word) {
            cnt += hw(word);
        }
    }
    return cnt;
}
u64 DenseSet::__len__() const {
    return get_weight();
}

vector<u64> DenseSet::get_counts_by_weight() const {
    vector<u64> res(n+1);
    auto func = [&] (u64 v) -> void { res[hw(v)] += 1; };
    iter_support(func);
    return res;
}

// ========================================
// Main methods
// ========================================
void DenseSet::do_Mobius(u64 mask) {
    do_Sweep<XOR_up<u64>>(mask);
}
void DenseSet::do_Complement() {
    u64 mask = n >= 6 ? -1ull : ((1ull << (1ull << n)) - 1ull);
    fori (i, data.size()) {
        data[i] = mask ^ data[i];
    }
}
void DenseSet::do_Not(u64 mask) {
    mask &= (1ull << n)-1;
    if (HI(mask) == 0) {
        fori (i, data.size()) {
            GenericSweepWord<SWAP<u64>>(data[i], mask); // LO(mask)
        }
    }
    else {
        fori (i, data.size()) {
            u64 j = i ^ HI(mask);
            ensure(j < data.size());
            if (j < u64(i))
                continue;
            GenericSweepWord<SWAP<u64>>(data[i], mask); // LO(mask)
            GenericSweepWord<SWAP<u64>>(data[j], mask); // LO(mask)
            swap(data[i], data[j]);
        }
    }
}
void DenseSet::do_UpperSet(u64 mask) {
    do_Sweep<OR_up<u64>>(mask);
}
void DenseSet::do_LowerSet(u64 mask) {
    do_Sweep<OR_down<u64>>(mask);
}
void DenseSet::do_MinSet(u64 mask) {
    do_UpperSet(mask);
    do_Sweep<LESS_up<u64>>(mask);
}
void DenseSet::do_MaxSet(u64 mask) {
    do_LowerSet(mask);
    do_Sweep<MORE_down<u64>>(mask);
}
void DenseSet::do_DivCore(u64 mask) {
    do_Sweep<XOR_up<u64>>(mask);
    do_MaxSet(mask);
    do_Not(mask);
}
void DenseSet::do_ComplementU2L(bool is_upper, u64 mask) {
    if (!is_upper)
        do_Sweep<OR_up<u64>>(mask);
    do_Complement();
    do_MaxSet(mask);
}
void DenseSet::do_ComplementL2U(bool is_lower, u64 mask) {
    if (!is_lower)
        do_Sweep<OR_down<u64>>(mask);
    do_Complement();
    do_MinSet(mask);
}
void DenseSet::do_UpperSet_Up1(bool is_minset, u64 mask) {
    if (!is_minset)
        do_MinSet(-1ull);

    // not in-place :( but min set should not be large
    auto minset = get_support();
    for (auto uv: minset) {
        data[uv >> 6] = 0;
    }
    for (auto uv: minset) {
         fori(i, n) {
            if ((mask & (1ull << i)) == 0)
                continue;
            u64 uv2 = uv | (1ull << i);
            if (uv2 != uv) {
                data[uv2 >> 6] |= 1ull << (uv2 & 0x3f);
            }
        }
    }
}

// ========================================
// Stuff
// ========================================
void DenseSet::save_to_file(const char *filename) const {
    fprintf(stderr, "Saving DenseSet(n=%d) to %s\n", n, filename);
    vector<u64> supp = get_support();

    FILE *fd = fopen(filename, "w");
    ensure(fd);

    u64 vn = n;
    u64 vl = supp.size();
    fwrite(&vn, 8, 1, fd);
    fwrite(&vl, 8, 1, fd);
    for (auto v: supp) {
        fwrite(&v, 8, 1, fd);
    }
    fclose(fd);
}
DenseSet DenseSet::load_from_file(const char *filename) {
    FILE *fd = fopen(filename, "r");
    // ensure(fd);

    u64 vl;
    u64 vn;
    fread(&vn, 8, 1, fd);
    fread(&vl, 8, 1, fd);

    fprintf(stderr, "Loading DenseSet(n=%lu) with weight %lu from %s\n", vn, vl, filename);

    DenseSet res(vn);
    fori (i, vl) {
        u64 v;
        fread(&v, 8, 1, fd);
        res.set(v);
    }
    fclose(fd);
    return res;
}

u64 DenseSet::get_hash() const {
    u64 h = -1ull;
    for (auto v: data) {
        h ^= v;
        h *= 0xcaffee1234abcdefull;
        h ^= h >> 12;
        h += v;
        h ^= h >> 17;
    }
    return h;
}
std::string DenseSet::info(const char *name) const {
    string sname = name ? name : "?";
    char buf[4096];
    snprintf(
        buf, 4000,
        "%016lx:%s n=%d wt=%lu |",
        get_hash(), sname.c_str(), n, get_weight()
    );

    string ret = buf;

    auto by_wt = get_counts_by_weight();
    fori (i, n+1) {
        if (by_wt[i]) {
            snprintf(buf, 4000, " %lu:%lu", i, by_wt[i]);
            ret += buf;
        }
    };
    return ret;
}
std::string DenseSet::__str__() const {
    return info(NULL);
}
void DenseSet::log_info(const char *name) const {
    fprintf(stderr, "%s\n", info(name).c_str(), "\n");
}

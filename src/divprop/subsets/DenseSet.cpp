#include "common.hpp"

#include "Sweep.hpp"

#include "DenseSet.hpp"

bool DenseSet::QUIET = false;

void DenseSet::set_quiet(bool value) {
    DenseSet::QUIET = value;
}

static uint64_t __get_lo_mask(int n) {
    ensure(n >= 0);
    if (n >= 6) {
        return -1ull;
    }
    return (1ull << (1 << n)) - 1;
}

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
    if (n == 0) {
        data.clear();
        return;
    }
    data.assign(HICEIL(1ull << n), 0);
}
void DenseSet::fill() {
    if (n == 0) {
        return;
    }
    data.assign(HICEIL(1ull << n), -1ull);
    if (n <= 6) {
        data[0] &= __get_lo_mask(n);
    }
}


bool DenseSet::is_empty() const {
    return get_weight() == 0;
}
bool DenseSet::__bool__() const {
    return !is_empty();
}
bool DenseSet::is_full() const {
    return get_weight() == 1ull << n;
}

// ========================================
// Single bit get/set
// ========================================
int DenseSet::get(uint64_t x) const {
    // ensure(x < 1ull << n);
    return (data[HI(x)] >> LO(x)) & 1;
}
bool DenseSet::__contains__(uint64_t x) const {
    return get(x) == 1;
}
void DenseSet::set(uint64_t x) {
    data[HI(x)] |= 1ull << LO(x);
}
void DenseSet::set(uint64_t x, uint64_t value) {
    data[HI(x)] &= ~(1ull << LO(x));
    data[HI(x)] |= (value & 1ull) << LO(x);
}

// ========================================
// Tools
// ========================================
template<auto func>
void DenseSet::do_Sweep(uint64_t mask) {
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
    do_Sweep<OR_up<uint64_t>>(mask);
}
void DenseSet::do_Sweep_OR_down(uint64_t mask) {
    do_Sweep<OR_down<uint64_t>>(mask);
}
void DenseSet::do_Sweep_XOR_up(uint64_t mask) {
    do_Sweep<XOR_up<uint64_t>>(mask);
}
void DenseSet::do_Sweep_XOR_down(uint64_t mask) {
    do_Sweep<XOR_down<uint64_t>>(mask);
}
void DenseSet::do_Sweep_AND_up(uint64_t mask) {
    do_Sweep<AND_up<uint64_t>>(mask);
}
void DenseSet::do_Sweep_AND_down(uint64_t mask) {
    do_Sweep<AND_down<uint64_t>>(mask);
}
void DenseSet::do_Sweep_SWAP(uint64_t mask) {
    do_Sweep<SWAP<uint64_t>>(mask);
}
void DenseSet::do_Sweep_LESS_up(uint64_t mask) {
    do_Sweep<LESS_up<uint64_t>>(mask);
}
void DenseSet::do_Sweep_MORE_down(uint64_t mask) {
    do_Sweep<MORE_down<uint64_t>>(mask);
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
    bool not_equal = 0;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        not_equal = 1;
        if ((data[i] & data2[i]) != data[i]) return false; // prec
    }
    return not_equal;
}
bool DenseSet::operator>(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    bool not_equal = 0;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        not_equal = 1;
        if ((data[i] & data2[i]) != data2[i]) return false; // succ
    }
    return not_equal;
}
bool DenseSet::operator<=(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        if ((data[i] & data2[i]) != data[i]) return false;
    }
    return true;
}
bool DenseSet::operator>=(const DenseSet & b) const {
    ensure(is_compatible_set(b));
    auto &data2 = b.data;
    fori(i, data.size()) {
        if (data[i] == data2[i]) continue;
        if ((data[i] & data2[i]) != data2[i]) return false; // succ
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
DenseSet DenseSet::operator|(const DenseSet & b) const {
    auto res = copy();
    res |= b;
    return res;
}
DenseSet DenseSet::operator^(const DenseSet & b) const {
    auto res = copy();
    res ^= b;
    return res;
}
DenseSet DenseSet::operator&(const DenseSet & b) const {
    auto res = copy();
    res &= b;
    return res;
}
DenseSet DenseSet::operator-(const DenseSet & b) const {
    auto res = copy();
    res -= b;
    return res;
}
DenseSet DenseSet::operator~() const {
    return Complement();
}

DenseSet DenseSet::get_head_fixed(int h, uint64_t value) {
    ensure(value < (1ull << h));
    ensure(h >=0 && h <= n);
    ensure(n - h >= 6);
    DenseSet result(n - h);
    uint64_t start = value << (n - h - 6);
    uint64_t end = (value + 1) << (n - h - 6);
    result.data = vector<uint64_t>(data.begin() + start, data.begin() + end);
    return result;
}

// ========================================
// Support
// ========================================

void DenseSet::iter_support(function<void(uint64_t)> const & func) const {
    if (n >= 6) {
        fori (hi, data.size()) {
            if (data[hi]) {
                fori (lo, 64) {
                    if ((data[hi] >> lo) & 1) {
                        uint64_t x = (hi << 6) | lo;
                        func(x);
                    }
                }
            }
        }
    }
    else {
        auto word = data[0];
        word &= __get_lo_mask(n);
        fori (lo, 64) {
            if ((word >> lo) & 1) {
                func((uint64_t)lo);
            }
        }
    }
}

// returns support of the function
// 32-bit version useful? if support is small then 2x RAM does not matter
// if it's large then working with DenseSet is better anyway...
vector<uint64_t> DenseSet::get_support() const {
    vector<uint64_t> inds;
    auto func = [&] (uint64_t v) -> void { inds.push_back(v); };
    iter_support(func);
    return inds;
}

uint64_t DenseSet::get_weight() const {
    uint64_t cnt = 0;
    if (n >= 6) {
        for (auto word: data) {
            if (word) {
                cnt += hw(word);
            }
        }
    }
    else {
        return hw(data[0] & __get_lo_mask(n));
    }
    return cnt;
}
uint64_t DenseSet::__len__() const {
    return get_weight();
}

vector<uint64_t> DenseSet::get_counts_by_weights() const {
    vector<uint64_t> res(n+1);
    auto func = [&] (uint64_t v) -> void { res[hw(v)] += 1; };
    iter_support(func);
    return res;
}
map<pair<int,int>,uint64_t> DenseSet::get_counts_by_weight_pairs(int n1, int n2) const {
    ensure(n == n1 + n2);

    map<pair<int,int>, uint64_t> res;
    uint64_t mask2 = (1ull << n2)-1;
    auto func = [&] (uint64_t v) -> void {
        uint64_t l = v >> n2;
        uint64_t r = v &mask2;
        res[make_pair(hw(l),hw(r))] += 1;
    };
    iter_support(func);
    return res;
}
string DenseSet::str_stat_by_weights() const {
    string ret;
    char buf[4096] = "";
    auto by_wt = get_counts_by_weights();
    fori (i, n+1) {
        if (by_wt[i]) {
            snprintf(buf, 4000, "%lu:%lu ", i, by_wt[i]);
            ret += buf;
        }
    };
    if (ret.size()) {
        ret.erase(ret.end() - 1);
    }
    return ret;
}
string DenseSet::str_stat_by_weight_pairs(int n1, int n2) const {
    string ret;
    char buf[4096] = "";
    auto by_wt = get_counts_by_weight_pairs(n1, n2);
    for (auto &p: by_wt) {
        if (p.second) {
            snprintf(
                buf, 4000, "%d,%d:%lu ",
                p.first.first, p.first.second, p.second
            );
            ret += buf;
        }
    };
    if (ret.size()) {
        ret.erase(ret.end() - 1);
    }
    return ret;
}

// ========================================
// Main methods
// ========================================

void DenseSet::do_UnsetUp(uint64_t mask) {
    do_Sweep<ZERO_up<uint64_t>>(mask);
}
void DenseSet::do_UnsetDown(uint64_t mask) {
    do_Sweep<ZERO_down<uint64_t>>(mask);
}
void DenseSet::do_SetUp(uint64_t mask) {
    do_Sweep<ONE_up<uint64_t>>(mask);
}
void DenseSet::do_SetDown(uint64_t mask) {
    do_Sweep<ONE_down<uint64_t>>(mask);
}
void DenseSet::do_Mobius(uint64_t mask) {
    do_Sweep<XOR_up<uint64_t>>(mask);
}
void DenseSet::do_Complement() {
    uint64_t mask = n >= 6 ? -1ull : ((1ull << (1ull << n)) - 1ull);
    fori (i, data.size()) {
        data[i] = mask ^ data[i];
    }
}
void DenseSet::do_Not(uint64_t mask) {
    mask &= (1ull << n)-1;
    if (HI(mask) == 0) {
        fori (i, data.size()) {
            GenericSweepWord<SWAP<uint64_t>>(data[i], mask); // LO(mask)
        }
    }
    else {
        fori (i, data.size()) {
            uint64_t j = i ^ HI(mask);
            ensure(j < data.size());
            if (j < uint64_t(i))
                continue;
            GenericSweepWord<SWAP<uint64_t>>(data[i], mask); // LO(mask)
            GenericSweepWord<SWAP<uint64_t>>(data[j], mask); // LO(mask)
            swap(data[i], data[j]);
        }
    }
}
void DenseSet::do_UpperSet(uint64_t mask) {
    do_Sweep<OR_up<uint64_t>>(mask);
}
void DenseSet::do_LowerSet(uint64_t mask) {
    do_Sweep<OR_down<uint64_t>>(mask);
}
void DenseSet::do_MinSet(uint64_t mask) {
    do_UpperSet(mask);
    do_Sweep<LESS_up<uint64_t>>(mask);
}
void DenseSet::do_MaxSet(uint64_t mask) {
    do_LowerSet(mask);
    do_Sweep<MORE_down<uint64_t>>(mask);
}
void DenseSet::do_DivCore(uint64_t mask) {
    do_Sweep<XOR_up<uint64_t>>(mask);
    do_MaxSet(mask);
    do_Not(mask);
}
void DenseSet::do_ComplementU2L(bool is_upper, uint64_t mask) {
    if (!is_upper)
        do_Sweep<OR_up<uint64_t>>(mask);
    do_Complement();
    do_MaxSet(mask);
}
void DenseSet::do_ComplementL2U(bool is_lower, uint64_t mask) {
    if (!is_lower)
        do_Sweep<OR_down<uint64_t>>(mask);
    do_Complement();
    do_MinSet(mask);
}
void DenseSet::do_UpperSet_Up1(bool is_minset, uint64_t mask) {
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
            uint64_t uv2 = uv | (1ull << i);
            if (uv2 != uv) {
                data[uv2 >> 6] |= 1ull << (uv2 & 0x3f);
            }
        }
    }
}

DenseSet DenseSet::Mobius(uint64_t mask) const {
    auto ret = copy();
    ret.do_Mobius(mask);
    return ret;
}
DenseSet DenseSet::Complement() const {
    auto ret = copy();
    ret.do_Complement();
    return ret;
}
DenseSet DenseSet::Not(uint64_t mask) const {
    auto ret = copy();
    ret.do_Not(mask);
    return ret;
}
DenseSet DenseSet::UpperSet(uint64_t mask) const {
    auto ret = copy();
    ret.do_UpperSet(mask);
    return ret;
}
DenseSet DenseSet::LowerSet(uint64_t mask) const {
    auto ret = copy();
    ret.do_LowerSet(mask);
    return ret;
}
DenseSet DenseSet::MinSet(uint64_t mask) const {
    auto ret = copy();
    ret.do_MinSet(mask);
    return ret;
}
DenseSet DenseSet::MaxSet(uint64_t mask) const {
    auto ret = copy();
    ret.do_MaxSet(mask);
    return ret;
}
DenseSet DenseSet::DivCore(uint64_t mask) const {
    auto ret = copy();
    ret.do_DivCore(mask);
    return ret;
}
DenseSet DenseSet::ComplementU2L(bool is_upper, uint64_t mask) const {
    auto ret = copy();
    ret.do_ComplementU2L(is_upper, mask);
    return ret;
}
DenseSet DenseSet::ComplementL2U(bool is_lower, uint64_t mask) const {
    auto ret = copy();
    ret.do_ComplementL2U(is_lower, mask);
    return ret;
}
DenseSet DenseSet::UpperSet_Up1(bool is_minset, uint64_t mask) const {
    auto ret = copy();
    ret.do_UpperSet_Up1(is_minset, mask);
    return ret;
}

// ========================================
// Stuff
// ========================================
void DenseSet::save_to_file(const char *filename) const {
    if (!QUIET) {
        fprintf(stderr, "Saving DenseSet(n=%d) to %s\n", n, filename);
    }
    vector<uint64_t> supp = get_support();

    FILE *fd = fopen(filename, "w");
    ensure(fd);

    uint64_t header = DenseSet::VERSION1;
    uint64_t vn = n;
    uint64_t vl = supp.size();
    uint64_t sz = 8;
    if (n <= 8) {
        sz = 1;
    }
    else if (n <= 16) {
        sz = 2;
    }
    else if (n <= 32) {
        sz = 4;
    }
    fwrite(&header, 8, 1, fd);
    fwrite(&vn, 8, 1, fd);
    fwrite(&vl, 8, 1, fd);
    fwrite(&sz, 8, 1, fd);

    for (auto v: supp) {
        fwrite(&v, sz, 1, fd);
    }

    uint64_t marker = MARKER_END;
    fwrite(&marker, 8, 1, fd);
    fclose(fd);
}
DenseSet DenseSet::load_from_file(const char *filename) {
    FILE *fd = fopen(filename, "r");
    // ensure(fd);

    DenseSet res;

    uint64_t header;
    fread(&header, 8, 1, fd);

    if (header == VERSION1) {
        uint64_t vl;
        uint64_t vn;
        uint64_t sz;
        fread(&vn, 8, 1, fd);
        fread(&vl, 8, 1, fd);
        fread(&sz, 8, 1, fd);

        if (!QUIET) {
            fprintf(stderr,
                "Loading DenseSet(n=%lu)"
                " with weight %lu from %s"
                " (%lu bytes per elem.)\n",
                vn, vl, filename, sz
            );
        }

        res = DenseSet(vn);
        uint64_t v = 0;
        fori (i, vl) {
            fread(&v, sz, 1, fd);
            res.set(v);
        }

        uint64_t marker;
        fread(&marker, 8, 1, fd);
        ensure(marker == MARKER_END);

        fclose(fd);
    }
    else {
        ensure(0, "unknown version");
    }
    return res;
}

uint64_t DenseSet::get_hash() const {
    uint64_t h = -1ull;
    for (auto v: data) {
        h ^= v;
        h *= 0xcaffee1234abcdefull;
        h ^= h >> 12;
        h += v;
        h ^= h >> 17;
    }
    return h;
}
std::string DenseSet::info() const {
    char buf[4096];
    snprintf(
        buf, 4000,
        "%016lx n=%d wt=%lu | ",
        get_hash(), n, get_weight()
    );
    return string(buf) + str_stat_by_weights();
}
std::string DenseSet::__str__() const {
    return info();
}
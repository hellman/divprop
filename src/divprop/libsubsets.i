%module(package="divprop") libsubsets

%include <std_vector.i>
%include <std_string.i>
%include <exception.i>
%include <std_map.i>
%include <std_unordered_set.i>
%include <stdint.i>

%typedef uint64_t u64;
%typedef uint32_t u32;
%typedef uint16_t u16;
%typedef uint8_t u8;

%typedef int64_t i64;
%typedef int32_t i32;
%typedef int16_t i16;
%typedef int8_t i8;

%template(MyVector_u64) std::vector<uint64_t>;
%template(MyVector_u32) std::vector<uint32_t>;
%template(MyVector_u16) std::vector<uint16_t>;
%template(MyVector_u8) std::vector<uint8_t>;
%template(MyVector_int) std::vector<int>;
%template(MyMap_PII_u64) std::map<std::pair<int,int>, uint64_t>;

// https://stackoverflow.com/questions/1394484/how-do-i-propagate-c-exceptions-to-python-in-a-swig-wrapper-library
%exception {
    try {
        $action
    } catch(const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (...) {
        SWIG_exception(SWIG_UnknownError, "unknown exception");
    }
}

%include "subsets/DenseSet.hpp"
%include "subsets/SboxGraph.hpp"
%include "divprop/DivCore.hpp"

%template(Sbox2GI) Sbox2GraphIndicator<uint64_t>;

%{
#include "DenseSet.hpp"
#include "SboxGraph.hpp"
#include "DivCore.hpp"
%}

%template(DivCore_StrongComposition8) tpl_DivCore_StrongComposition<uint8_t>;
%template(DivCore_StrongComposition16) tpl_DivCore_StrongComposition<uint16_t>;
%template(DivCore_StrongComposition32) tpl_DivCore_StrongComposition<uint32_t>;
// 64-bit very hardly useful...
%template(DivCore_StrongComposition64) tpl_DivCore_StrongComposition<uint64_t>;

%pythoncode %{
DivCore_StrongComposition = DivCore_StrongComposition32;
%}

%template(MyVector_DenseSet) std::vector<DenseSet>;
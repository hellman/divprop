%module(package="divprop") divprop

%include <std_vector.i>
%include <std_string.i>
%include <exception.i>
%include <std_map.i>
%include <std_set.i>
%include <stdint.i>

%template(MyVector_u64) std::vector<uint64_t>;
%template(MyVector_u32) std::vector<uint32_t>;
%template(MyVector_u16) std::vector<uint16_t>;
%template(MyVector_u8) std::vector<uint8_t>;
%template(MyVector_int) std::vector<int>;
%template(MySet_uint64) std::set<uint64_t>;
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

%{
#include "DenseSet.hpp"
#include "DivCore.hpp"
#include "Sbox.hpp"
%}

%pythoncode %{
	from subsets.subsets import DenseSet
%}

%include "DivCore.hpp"
%include "../sbox/Sbox.hpp"

%template(Sbox8) T_Sbox<uint8_t>;
%template(Sbox16) T_Sbox<uint16_t>;
%template(Sbox32) T_Sbox<uint32_t>;
%template(Sbox64) T_Sbox<uint64_t>;
%pythoncode %{
Sbox = Sbox64
Sbox.classes = Sbox8, Sbox16, Sbox32, Sbox64
%}

%template(DivCore_StrongComposition8) T_DivCore_StrongComposition<uint8_t>;
%template(DivCore_StrongComposition16) T_DivCore_StrongComposition<uint16_t>;
%template(DivCore_StrongComposition32) T_DivCore_StrongComposition<uint32_t>;
%template(DivCore_StrongComposition64) T_DivCore_StrongComposition<uint64_t>;
%pythoncode %{
DivCore_StrongComposition = DivCore_StrongComposition32
%}

%template(Vec_DenseSet) std::vector<DenseSet>;
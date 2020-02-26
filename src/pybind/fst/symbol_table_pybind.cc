// pybind/fst/symbol_table_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "fst/symbol_table_pybind.h"

#include "fst/symbol-table.h"

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

void pybind_symbol_table(py::module& m) {
  m.attr("kNoSymbol") = fst::kNoSymbol;

  {
    using PyClass = fst::SymbolTableReadOptions;
    py::class_<PyClass>(m, "SymbolTableReadOptions")
        .def(
            py::init<std::vector<std::pair<int64, int64>>, const fst::string>(),
            py::arg("string_hash_ranges"), py::arg("source"))
        .def_readwrite("string_hash_ranges", &PyClass::string_hash_ranges)
        .def_readwrite("source", &PyClass::source);
  }
  {
    using PyClass = fst::SymbolTableTextOptions;
    py::class_<PyClass>(m, "SymbolTableTextOptions")
        .def(py::init<bool>(), py::arg("allow_negative_labels") = false)
        .def_readwrite("allow_negative_labels", &PyClass::allow_negative_labels)
        .def_readwrite("fst_field_separator", &PyClass::fst_field_separator);
  }
  {
    using PyClass = fst::SymbolTable;
    py::class_<PyClass>(
        m, "SymbolTable",
        "Symbol (string) to integer (and reverse) mapping.\n"
        "\n"
        "The SymbolTable implements the mappings of labels to strings and "
        "reverse. SymbolTables are used to describe the alphabet of the input "
        "and output abels for arcs in a Finite State Transducer."
        "\n"
        "SymbolTables are reference-counted and can therefore be shared across "
        "multiple machines. For example a language model grammar G, with a "
        "SymbolTable for the words in the language model can share this symbol "
        "table with the lexical representation L o G.")
        .def(py::init<const fst::string&>(),
             "Constructs symbol table with an optional name.",
             py::arg("name") = "<unspecified>")
        .def_static("ReadText",
                    overload_cast_<const fst::string&,
                                   const fst::SymbolTableTextOptions&>()(
                        &PyClass::ReadText),
                    "Reads a text representation of the symbol table",
                    py::arg("filename"),
                    py::arg("opts") = fst::SymbolTableTextOptions(),
                    py::return_value_policy::take_ownership)
        .def_static(
            "Read",
            overload_cast_<std::istream&, const fst::SymbolTableReadOptions&>()(
                &PyClass::Read),
            "WARNING: Reading via symbol table read options should not be "
            "used. This is a temporary work-around.",
            py::arg("strm"), py::arg("opts") = fst::SymbolTableReadOptions())
        .def_static("Read", overload_cast_<std::istream&, const fst::string&>()(
                                &PyClass::Read),
                    "Reads a binary dump of the symbol table from a stream.",
                    py::arg("strm"), py::arg("source"))
        .def_static(
            "Read", overload_cast_<const fst::string&>()(&PyClass::Read),
            "Reads a binary dump of the symbol table.", py::arg("filename"))
        .def("Copy", &PyClass::Copy, "Creates a reference counted copy.")
        .def("AddSymbol",
             // clang-format off
             (int64 (PyClass::*)(const fst::string&, int64)) &PyClass::AddSymbol,
             // clang-format on
             "Adds a symbol with given key to table. A symbol table also keeps "
             "track of the last available key (highest key value in the symbol "
             "table).",
             py::arg("symbol"), py::arg("key"))
        .def("AddSymbol",
             (int64 (PyClass::*)(const fst::string&)) & PyClass::AddSymbol,
             "Adds a symbol to the table. The associated value key is "
             "automatically assigned by the symbol table.",
             py::arg("symbol"))
        .def("AddTable", &PyClass::AddTable,
             "Adds another symbol table to this table. All key values will be "
             "offset"
             "by the current available key (highest key value in the symbol "
             "table)."
             "Note string symbols with the same key value will still have the "
             "same"
             "key value after the symbol table has been merged, but a different"
             "value. Adding symbol tables do not result in changes in the base "
             "table.",
             py::arg("table"))
        .def("RemoveSymbol", &PyClass::RemoveSymbol, py::arg("key"))
        .def("Name", &PyClass::Name, "Returns the name of the symbol table.")
        .def("SetName", &PyClass::SetName, "Sets the name of the symbol table.")
        .def("CheckSum", &PyClass::CheckSum,
             "Return the label-agnostic MD5 check-sum for this table. All new "
             "symbols added to the table will result in an updated checksum. "
             "Deprecated.")
        .def("LabeledCheckSum", &PyClass::LabeledCheckSum,
             "Same as CheckSum(), but returns an label-dependent version.")
        .def("Write", (bool (PyClass::*)(std::ostream&) const) & PyClass::Write,
             py::arg("strm"))
        .def("Write",
             (bool (PyClass::*)(const fst::string&) const) & PyClass::Write,
             py::arg("filename"))
        .def("WriteText",
             // clang-format off
             (bool (PyClass::*)(std::ostream&, const fst::SymbolTableTextOptions&) const) &PyClass::WriteText,
             // clang-format on
             "Dump a text representation of the symbol table via a stream.",
             py::arg("strm"), py::arg("opts") = fst::SymbolTableTextOptions())
        .def("WriteText",
             (bool (PyClass::*)(const fst::string&) const) & PyClass::WriteText,
             "Dump a text representation of the symbol table.",
             py::arg("filename"))
        .def("Find", (fst::string (PyClass::*)(int64) const) & PyClass::Find,
             "Returns the string associated with the key; if the key is out of"
             "range (<0, >max), returns an empty string.",
             py::arg("key"))
        .def("Find",
             (int64 (PyClass::*)(const fst::string&) const) & PyClass::Find,
             "Returns the key associated with the symbol; if the symbol does "
             "not exist, kNoSymbol is returned.",
             py::arg("symbol"))
        .def("Find", (int64 (PyClass::*)(const char*) const) & PyClass::Find,
             "Returns the key associated with the symbol; if the symbol does "
             "not exist,"
             "kNoSymbol is returned.",
             py::arg("symbol"))
        .def("Member", (bool (PyClass::*)(int64) const) & PyClass::Member,
             py::arg("key"))
        .def("Member",
             (bool (PyClass::*)(const fst::string&) const) & PyClass::Member,
             py::arg("symbol"))
        .def("AvailableKey", &PyClass::AvailableKey,
             "Returns the current available key (i.e., highest key + 1) in the "
             "symbol table.")
        .def("NumSymbols", &PyClass::NumSymbols,
             "Returns the current number of symbols in table (not necessarily "
             "equal to AvailableKey()).")
        .def("GetNthKey", &PyClass::GetNthKey, py::arg("pos"))
        .def("__str__", [](const PyClass& sym) {
          std::ostringstream os;
          sym.WriteText(os);
          return os.str();
        });
  }
  {
    using PyClass = fst::SymbolTableIterator;
    py::class_<PyClass>(m, "SymbolTableIterator")
        .def(py::init<const fst::SymbolTable&>(), py::arg("table"))
        .def("Done", &PyClass::Done, "Returns whether iterator is done.")
        .def("Value", &PyClass::Value, "Return the key of the current symbol.")
        .def("Symbol", &PyClass::Symbol,
             "Return the string of the current symbol.")
        .def("Next", &PyClass::Next, "Advances iterator.")
        .def("Reset", &PyClass::Reset, "Resets iterator.");
  }
  m.def("RelabelSymbolTable", &fst::RelabelSymbolTable<int>,
        "Relabels a symbol table as specified by the input vector of pairs "
        "(old label, new label). The new symbol table only retains symbols for "
        "which a relabeling is explicitly specified.",
        py::arg("table"), py::arg("pairs"));

  m.def("CompatSymbols", &fst::CompatSymbols,
        "Returns true if the two symbol tables have equal checksums. Passing "
        "in nullptr for either table always returns true.",
        py::arg("sysm1"), py::arg("syms2"), py::arg("warning") = true);

  m.def("SymbolTableToString", &fst::SymbolTableToString, py::arg("table"),
        py::arg("result"));

  m.def("StringToSymbolTable", &fst::StringToSymbolTable, py::arg("str"),
        py::return_value_policy::take_ownership);
}

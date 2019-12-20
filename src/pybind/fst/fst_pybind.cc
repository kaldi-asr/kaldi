// pybind/fst/fst_pybind.cc

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

#include "fst/fst_pybind.h"

#include "fst/fst.h"

#include "fst/arc_pybind.h"
#include "fst/compile_pybind.h"
#include "fst/symbol_table_pybind.h"
#include "fst/vector_fst_pybind.h"
#include "fst/weight_pybind.h"
#include "fstext/kaldi_fst_io_pybind.h"

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

namespace {

void _pybind_fst(py::module& m) {
  m.attr("kNoLabel") = fst::kNoLabel;
  m.attr("kNoStateId") = fst::kNoStateId;
  {
    using PyClass = fst::FstHeader;
    py::class_<PyClass>(m, "FstHeader")
        .def(py::init<>())
        .def("FstType", &PyClass::FstType, py::return_value_policy::reference)
        .def("ArcType", &PyClass::ArcType, py::return_value_policy::reference)
        .def("Version", &PyClass::Version)
        .def("GetFlags", &PyClass::GetFlags)
        .def("Properties", &PyClass::Properties)
        .def("Start", &PyClass::Start)
        .def("NumStates", &PyClass::NumStates)
        .def("NumArcs", &PyClass::NumArcs)
        .def("SetFstType", &PyClass::SetFstType, py::arg("type"))
        .def("SetArcType", &PyClass::SetArcType, py::arg("type"))
        .def("SetVersion", &PyClass::SetVersion, py::arg("version"))
        .def("SetFlags", &PyClass::SetFlags, py::arg("flags"))
        .def("SetProperties", &PyClass::SetProperties, py::arg("properties"))
        .def("SetStart", &PyClass::SetStart, py::arg("start"))
        .def("SetNumStates", &PyClass::SetNumStates, py::arg("numstates"))
        .def("SetNumArcs", &PyClass::SetNumArcs, py::arg("numarcs"))
        .def("Read", &PyClass::Read, py::arg("strm"), py::arg("source"),
             py::arg("rewind") = false)
        .def("Write", &PyClass::Write, py::arg("strm"), py::arg("source"))
        .def("DebugString", &PyClass::DebugString);
  }
  {
    using PyClass = fst::FstWriteOptions;

    py::class_<PyClass>(m, "FstWriteOptions")
        .def_readwrite("source", &PyClass::source, "Where you're writing to.")
        .def_readwrite("write_header", &PyClass::write_header,
                       "Where you're writing to.")
        .def_readwrite("write_isymbols", &PyClass::write_isymbols,
                       "Write the header?")
        .def_readwrite("write_osymbols", &PyClass::write_osymbols,
                       "Write input symbols?")
        .def_readwrite("align", &PyClass::align,
                       "Write data aligned (may fail on pipes)?")
        .def_readwrite("stream_write", &PyClass::stream_write,
                       "Avoid seek operations in writing.")
        .def(
            py::init<const string&, bool, bool, bool, bool, bool>(),
            py::arg("source") = "<unspecified>", py::arg("write_header") = true,
            py::arg("write_isymbols") = true, py::arg("write_osymbols") = true,
            py::arg("align") = FLAGS_fst_align, py::arg("stream_write") = false)
        .def("__str__", [](const PyClass& opt) {
          std::ostringstream os;
          os << "source: " << opt.source << "\n"
             << "write_header: " << opt.write_header << "\n"
             << "write_isymbols: " << opt.write_isymbols << "\n"
             << "write_osymbols: " << opt.write_osymbols << "\n"
             << "align: " << opt.align << "\n"
             << "stream_write: " << opt.stream_write << "\n";
          return os.str();
        });
  }

  auto fst_read_options =
      py::class_<fst::FstReadOptions>(m, "FstReadOptions")
          .def(py::init<const fst::string&, const fst::FstHeader*,
                        const fst::SymbolTable*, const fst::SymbolTable*>(),
               py::arg("source") = "<unspecified>", py::arg("header") = nullptr,
               py::arg("isymbols") = nullptr, py::arg("osymbols") = nullptr)
          .def(py::init<const fst::string&, const fst::SymbolTable*,
                        const fst::SymbolTable*>(),
               py::arg("source"), py::arg("isymbols") = nullptr,
               py::arg("osymbols") = nullptr)
          .def_readwrite("source", &fst::FstReadOptions::source,
                         "Where you're reading from.")
          .def_readwrite("header", &fst::FstReadOptions::header,
                         "Pointer to FST header; if non-zero, use this info "
                         "(don't read a stream header).")
          .def_readwrite("isymbols", &fst::FstReadOptions::isymbols,
                         "Pointer to input symbols; if non-zero, use this info "
                         "(read and skip stream isymbols)")
          .def_readwrite("osymbols", &fst::FstReadOptions::osymbols,
                         "Pointer to output symbols; if non-zero, use this "
                         "info (read and skip stream osymbols)")
          .def_readwrite("mode", &fst::FstReadOptions::mode,
                         "Read or map files (advisory, if possible)")
          .def_readwrite("read_isymbols", &fst::FstReadOptions::read_isymbols,
                         "Read isymbols, if any (default: true).")
          .def_readwrite("read_osymbols", &fst::FstReadOptions::read_osymbols,
                         "Read osymbols, if any (default: true).")
          .def_static("ReadMode", &fst::FstReadOptions::ReadMode,
                      "Helper function to convert strings FileReadModes into "
                      "their enum value.",
                      py::arg("mode"))
          .def("DebugString", &fst::FstReadOptions::DebugString,
               "Outputs a debug string for the FstReadOptions object.");

  py::enum_<fst::FstReadOptions::FileReadMode>(
      fst_read_options, "FileReadMode", py::arithmetic(),
      "FileReadMode(s) are advisory, there are "
      "many conditions than prevent a\n"
      "file from being mapped, READ mode will "
      "be selected in these cases with\n"
      "a warning indicating why it was chosen.")
      .value("READ", fst::FstReadOptions::FileReadMode::READ)
      .value("MAP", fst::FstReadOptions::FileReadMode::MAP)
      .export_values();

  py::enum_<fst::MatchType>(m, "MatchType", py::arithmetic(),
                            "Specifies matcher action.")
      .value("MATCH_INPUT", fst::MatchType::MATCH_INPUT, "Match input label.")
      .value("MATCH_OUTPUT", fst::MatchType::MATCH_OUTPUT,
             "Match output label.")
      .value("MATCH_BOTH", fst::MatchType::MATCH_BOTH,
             "Match input or output label.")
      .value("MATCH_NONE", fst::MatchType::MATCH_NONE, "Match nothing.")
      .value("MATCH_UNKNOWN", fst::MatchType::MATCH_UNKNOWN,
             "match type unknown.")
      .export_values();
  {
    using PyClass = fst::StateIteratorBase<fst::StdArc>;
    py::class_<PyClass>(m, "StdArcStateIteratorBase")
        .def("Done", &PyClass::Done, "End of iterator?")
        .def("Value", &PyClass::Value, "Returns current state (when !Done()).")
        .def("Next", &PyClass::Next, "Advances to next state (when !Done()).")
        .def("Reset", &PyClass::Reset, "Resets to initial condition.");
  }

  {
    using PyClass = fst::StateIteratorData<fst::StdArc>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "StdArcStateIteratorData")
        .def(py::init<>())
        .def_readwrite("base", &PyClass::base,
                       "Specialized iterator if non-zero.")
        .def_readwrite("nstates", &PyClass::nstates,
                       "Otherwise, the total number of states.");
  }

  {
    using PyClass = fst::ArcIteratorBase<fst::StdArc>;
    py::class_<PyClass>(m, "StdArcArcIteratorBase")
        .def("Done", &PyClass::Done, "End of iterator?")
        .def("Value", &PyClass::Value, "Returns current arc (when !Done()).",
             py::return_value_policy::reference)
        .def("Next", &PyClass::Next, "Advances to next arc (when !Done()).")
        .def("Position", &PyClass::Position, "Returns current position.")
        .def("Reset", &PyClass::Reset, "Resets to initial condition.")
        .def("Seek", &PyClass::Seek, "Advances to arbitrary arc by position.")
        .def("Flags", &PyClass::Flags, "Returns current behavorial flags.")
        .def("SetFlags", &PyClass::SetFlags, "Sets behavorial flags.");
  }

  {
    using PyClass = fst::ArcIteratorData<fst::StdArc>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "StdArcArcIteratorData")
        .def(py::init<>())
        .def_readwrite("base", &PyClass::base,
                       "Specialized iterator if non-zero.")
        .def_readwrite("arcs", &PyClass::arcs, "Otherwise arcs pointer")
        .def_readwrite("narcs", &PyClass::narcs, "arc count")
        .def_readwrite("ref_count", &PyClass::ref_count,
                       "reference count if non-zero.");
  }

  {
    using PyClass = fst::StdFst;
    using Arc = PyClass::Arc;
    using StateId = PyClass::StateId;
    using Weight = PyClass::Weight;

    auto fst_state_iterator =
        py::class_<fst::StateIterator<fst::StdFst>>(m, "StdFstStateIterator");
    auto fst_arc_iterator =
        py::class_<fst::ArcIterator<fst::StdFst>>(m, "StdFstArcIterator");

    py::class_<PyClass>(
        m, "StdFst",
        "A generic FST, templated on the arc definition, with \n"
        "common-demoninator methods (use StateIterator and \n"
        "ArcIterator to iterate over its states and arcs).")
        .def("Start", &PyClass::Start, "Initial state.")
        .def("Final", &PyClass::Final, "State's final weight.")
        .def("NumArcs", &PyClass::NumArcs, "State's arc count.")
        .def("NumInputEpsilons", &PyClass::NumInputEpsilons,
             "State's output epsilon count.")
        .def("Properties", &PyClass::Properties,
             "Property bits. If test = false, return stored properties bits "
             "for mask\n"
             "(some possibly unknown); if test = true, return property bits "
             "for mask\n"
             "(computing o.w. unknown).",
             py::arg("mask"), py::arg("test"))
        .def("Type", &PyClass::Type, "FST typename",
             py::return_value_policy::reference)
        .def(
            "Copy", &PyClass::Copy,
            "Gets a copy of this Fst. The copying behaves as follows:\n",
            "\n"
            "(1) The copying is constant time if safe = false or if safe = "
            "true and is on an otherwise unaccessed FST.\n"
            "\n"
            "(2) If safe = true, the copy is thread-safe in that the original\n"
            "and copy can be safely accessed (but not necessarily mutated) by\n"
            "separate threads. For some FST types, 'Copy(true)' should only\n"
            "be called on an FST that has not otherwise been accessed.\n"
            "Behavior is otherwise undefined.\n"
            "\n"
            "(3) If a MutableFst is copied and then mutated, then the original"
            "\n"
            "is unmodified and vice versa (often by a copy-on-write on the \n"
            "initial mutation, which may not be constant time).",
            py::arg("safe") = false, py::return_value_policy::take_ownership)
        .def_static(
            "Read",
            // clang-format off
            overload_cast_<std::istream&, const fst::FstReadOptions&>()(&PyClass::Read),
            // clang-format on
            "Reads an FST from an input stream; returns nullptr on error.",
            py::arg("strm"), py::arg("opts"),
            py::return_value_policy::take_ownership)
        .def_static(
            "Read", overload_cast_<const fst::string&>()(&PyClass::Read),
            "Reads an FST from a file; returns nullptr on error. An empty\n"
            "filename results in reading from standard input.",
            py::arg("filename"), py::return_value_policy::take_ownership)
        .def("Write",
             // clang-format off
            (bool (PyClass::*)(std::ostream&, const fst::FstWriteOptions&)const)&PyClass::Write,
             // clang-format on
             "Writes an FST to an output stream; returns false on error.",
             py::arg("strm"), py::arg("opts"))
        .def("Write",
             (bool (PyClass::*)(const fst::string&) const) & PyClass::Write,
             "Writes an FST to a file; returns false on error; an empty\n"
             "filename results in writing to standard output.",
             py::arg("filename"))
        .def("InputSymbols", &PyClass::InputSymbols,
             "Returns input label symbol table; return nullptr if not "
             "specified.",
             py::return_value_policy::reference)
        .def("OutputSymbols", &PyClass::OutputSymbols,
             "Returns output label symbol table; return nullptr if not "
             "specified.",
             py::return_value_policy::reference)
        .def("InitStateIterator", &PyClass::InitStateIterator,
             "For generic state iterator construction (not normally called "
             "directly by users). Does not copy the FST.",
             py::arg("data"))
        .def("InitArcIterator", &PyClass::InitArcIterator,
             "For generic arc iterator construction (not normally called "
             "directly by users). Does not copy the FST.",
             py::arg("s"), py::arg("data"))
#if 0
      // TODO(fangjun): what is the use of InitMatcher?
        .def("InitMatcher", &PyClass::InitMatcher,
             "For generic matcher construction (not normally called directly "
             "by users).",
             py::arg("match_type")) // TODO(fangjun): reference semantics ?
#endif
        ;
    fst_state_iterator.def(py::init<const PyClass&>(), py::arg("fst"))
        .def("Done", &fst::StateIterator<PyClass>::Done)
        .def("Value", &fst::StateIterator<PyClass>::Value)
        .def("Next", &fst::StateIterator<PyClass>::Next)
        .def("Reset", &fst::StateIterator<PyClass>::Reset);

    fst_arc_iterator
        .def(py::init<const PyClass&, StateId>(), py::arg("fst"), py::arg("s"))
        .def("Done", &fst::ArcIterator<PyClass>::Done)
        .def("Value", &fst::ArcIterator<PyClass>::Value,
             py::return_value_policy::reference)
        .def("Next", &fst::ArcIterator<PyClass>::Next)
        .def("Reset", &fst::ArcIterator<PyClass>::Reset)
        .def("Seek", &fst::ArcIterator<PyClass>::Seek, py::arg("a"))
        .def("Position", &fst::ArcIterator<PyClass>::Position)
        .def("Flags", &fst::ArcIterator<PyClass>::Flags)
        .def("SetFlags", &fst::ArcIterator<PyClass>::SetFlags);
    ;
  }

  m.def("TestProperties", &fst::TestProperties<fst::StdArc>, py::arg("fst"),
        py::arg("mask"), py::arg("known"));

  m.def("FstToString",
        // clang-format off
      (fst::string (*)(const fst::StdFst&, const fst::FstWriteOptions&))&fst::FstToString<fst::StdArc>,
        // clang-format on
        py::arg("fst"),
        py::arg("options") = fst::FstWriteOptions("FstToString"));

  m.def("FstToString",
        // clang-format off
      (void (*)(const fst::StdFst&, fst::string*))&fst::FstToString<fst::StdArc>,
        // clang-format on
        py::arg("fst"), py::arg("result"));

  m.def("FstToString",
        // clang-format off
      (void (*)(const fst::StdFst&, fst::string*, const fst::FstWriteOptions&))&fst::FstToString<fst::StdArc>,
        // clang-format on
        py::arg("fst"), py::arg("result"), py::arg("options"));

  m.def("StringToFst", &fst::StringToFst<fst::StdArc>, py::arg("s"));
}

}  // namespace

void pybind_fst(py::module& _m) {
  py::module m = _m.def_submodule("fst", "FST pybind for Kaldi");

  // WARNING(fangjun): do NOT sort the following in alphabetic order!
  pybind_weight(m);
  pybind_arc(m);
  pybind_symbol_table(m);

  _pybind_fst(m);
  pybind_vector_fst(m);
  pybind_kaldi_fst_io(m);
  pybind_compile(m);
}

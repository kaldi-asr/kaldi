// pybind/fst/fst_pybind.cc

// Copyright 2019-2020   Mobvoi AI Lab, Beijing, China
//                       (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

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
#include "fstext/fstext_pybind.h"

#include "lat/determinize_lattice_pruned_pybind.h"

namespace {

void _pybind_fst(py::module& m) {
  m.attr("kNoLabel") = fst::kNoLabel;
  m.attr("kNoStateId") = fst::kNoStateId;
  m.attr("kDelta") = fst::kDelta;  // for weight.Quantize()

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

  pybind_state_iterator_base_impl<fst::StdArc>(m, "StdArcStateIteratorBase");
  pybind_state_iterator_data_impl<fst::StdArc>(m, "StdArcStateIteratorData");
  pybind_arc_iterator_base_impl<fst::StdArc>(m, "StdArcArcIteratorBase");
  pybind_arc_iterator_data_impl<fst::StdArc>(m, "StdArcArcIteratorData");
  pybind_fst_impl<fst::StdArc>(
      m, "StdFst",
      "A generic FST, templated on the arc definition, with \n"
      "common-demoninator methods (use StateIterator and \n"
      "ArcIterator to iterate over its states and arcs).");
  pybind_state_iterator_impl<fst::StdFst>(m, "StdFstStateIterator");
  pybind_arc_iterator_impl<fst::StdFst>(m, "StdFstArcIterator");

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
  pybind_compile(m);

  pybind_fstext(m);

  pybind_determinize_lattice_pruned(m);
}

// pybind/fst/vector_fst_pybind.cc

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

#include "fst/vector_fst_pybind.h"

#include "fst/script/info-impl.h"

#include "fst/script/fst-class.h"
#include "fst/script/print-impl.h"
#include "fst/vector-fst.h"

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

namespace {
// this following function is copied from openfst/src/script/info-impl.cc
void PrintFstInfoImpl(const fst::FstInfo& fstinfo, std::ostream& ostrm) {
  using namespace fst;
  ostrm.setf(std::ios::left);
  ostrm.width(50);
  ostrm << "fst type" << fstinfo.FstType() << std::endl;
  ostrm.width(50);
  ostrm << "arc type" << fstinfo.ArcType() << std::endl;
  ostrm.width(50);
  ostrm << "input symbol table" << fstinfo.InputSymbols() << std::endl;
  ostrm.width(50);
  ostrm << "output symbol table" << fstinfo.OutputSymbols() << std::endl;
  if (!fstinfo.LongInfo()) {
    return;
  }
  ostrm.width(50);
  ostrm << "# of states" << fstinfo.NumStates() << std::endl;
  ostrm.width(50);
  ostrm << "# of arcs" << fstinfo.NumArcs() << std::endl;
  ostrm.width(50);
  ostrm << "initial state" << fstinfo.Start() << std::endl;
  ostrm.width(50);
  ostrm << "# of final states" << fstinfo.NumFinal() << std::endl;
  ostrm.width(50);
  ostrm << "# of input/output epsilons" << fstinfo.NumEpsilons() << std::endl;
  ostrm.width(50);
  ostrm << "# of input epsilons" << fstinfo.NumInputEpsilons() << std::endl;
  ostrm.width(50);
  ostrm << "# of output epsilons" << fstinfo.NumOutputEpsilons() << std::endl;
  ostrm.width(50);
  ostrm << "input label multiplicity" << fstinfo.InputLabelMultiplicity()
        << std::endl;
  ostrm.width(50);
  ostrm << "output label multiplicity" << fstinfo.OutputLabelMultiplicity()
        << std::endl;
  ostrm.width(50);
  string arc_type = "";
  if (fstinfo.ArcFilterType() == "epsilon")
    arc_type = "epsilon ";
  else if (fstinfo.ArcFilterType() == "iepsilon")
    arc_type = "input-epsilon ";
  else if (fstinfo.ArcFilterType() == "oepsilon")
    arc_type = "output-epsilon ";
  const auto accessible_label = "# of " + arc_type + "accessible states";
  ostrm.width(50);
  ostrm << accessible_label << fstinfo.NumAccessible() << std::endl;
  const auto coaccessible_label = "# of " + arc_type + "coaccessible states";
  ostrm.width(50);
  ostrm << coaccessible_label << fstinfo.NumCoAccessible() << std::endl;
  const auto connected_label = "# of " + arc_type + "connected states";
  ostrm.width(50);
  ostrm << connected_label << fstinfo.NumConnected() << std::endl;
  const auto numcc_label = "# of " + arc_type + "connected components";
  ostrm.width(50);
  ostrm << numcc_label << fstinfo.NumCc() << std::endl;
  const auto numscc_label = "# of " + arc_type + "strongly conn components";
  ostrm.width(50);
  ostrm << numscc_label << fstinfo.NumScc() << std::endl;
  ostrm.width(50);
  ostrm << "input matcher"
        << (fstinfo.InputMatchType() == MATCH_INPUT
                ? 'y'
                : fstinfo.InputMatchType() == MATCH_NONE ? 'n' : '?')
        << std::endl;
  ostrm.width(50);
  ostrm << "output matcher"
        << (fstinfo.OutputMatchType() == MATCH_OUTPUT
                ? 'y'
                : fstinfo.OutputMatchType() == MATCH_NONE ? 'n' : '?')
        << std::endl;
  ostrm.width(50);
  ostrm << "input lookahead" << (fstinfo.InputLookAhead() ? 'y' : 'n')
        << std::endl;
  ostrm.width(50);
  ostrm << "output lookahead" << (fstinfo.OutputLookAhead() ? 'y' : 'n')
        << std::endl;
  uint64 prop = 1;
  for (auto i = 0; i < 64; ++i, prop <<= 1) {
    if (prop & kBinaryProperties) {
      char value = 'n';
      if (fstinfo.Properties() & prop) value = 'y';
      ostrm.width(50);
      ostrm << PropertyNames[i] << value << std::endl;
    } else if (prop & kPosTrinaryProperties) {
      char value = '?';
      if (fstinfo.Properties() & prop)
        value = 'y';
      else if (fstinfo.Properties() & prop << 1)
        value = 'n';
      ostrm.width(50);
      ostrm << PropertyNames[i] << value << std::endl;
    }
  }
}
}

void pybind_vector_fst(py::module& m) {
  {
    using PyClass = fst::StdVectorFst;
    using Arc = PyClass::Arc;
    using StateId = PyClass::StateId;
    using State = PyClass::State;

    py::class_<PyClass>(m, "StdVectorFst")
        .def(py::init<>())
        .def(py::init<const fst::StdFst&>(), py::arg("fst"))
        .def(py::init<const PyClass&, bool>(), py::arg("fst"),
             py::arg("safe") = false)
        .def("Start", &PyClass::Start)
        .def("Final", &PyClass::Final, py::arg("s"))
        .def("SetStart", &PyClass::SetStart, py::arg("s"))
        .def("SetFinal", &PyClass::SetFinal, py::arg("s"), py::arg("weight"))
        .def("SetProperties", &PyClass::SetProperties, py::arg("props"),
             py::arg("mask"))
        .def("AddState", (StateId (PyClass::*)()) & PyClass::AddState)
        .def("AddArc", &PyClass::AddArc, py::arg("s"), py::arg("arc"))
        .def("DeleteStates", (void (PyClass::*)(const std::vector<StateId>&)) &
                                 PyClass::DeleteStates,
             py::arg("dstates"))
        .def("DeleteStates", (void (PyClass::*)()) & PyClass::DeleteStates,
             "Delete all states")
        .def("DeleteArcs",
             (void (PyClass::*)(StateId, size_t)) & PyClass::DeleteArcs,
             py::arg("state"), py::arg("n"))
        .def("DeleteArcs", (void (PyClass::*)(StateId)) & PyClass::DeleteArcs,
             py::arg("s"))
        .def("ReserveStates", &PyClass::ReserveStates, py::arg("s"))
        .def("ReserveArcs", &PyClass::ReserveArcs, py::arg("s"), py::arg("n"))
        .def("InputSymbols", &PyClass::InputSymbols,
             "Returns input label symbol table; return nullptr if not "
             "specified.",
             py::return_value_policy::reference)
        .def("OutputSymbols", &PyClass::OutputSymbols,
             "Returns output label symbol table; return nullptr if not "
             "specified.",
             py::return_value_policy::reference)
        .def("MutableInputSymbols", &PyClass::MutableInputSymbols,
             "Returns input label symbol table; return nullptr if not "
             "specified.",
             py::return_value_policy::reference)
        .def("MutableOutputSymbols", &PyClass::MutableOutputSymbols,
             "Returns output label symbol table; return nullptr if not "
             "specified.",
             py::return_value_policy::reference)
        .def("SetInputSymbols", &PyClass::SetInputSymbols, py::arg("isyms"))
        .def("SetOutputSymbols", &PyClass::SetOutputSymbols, py::arg("osyms"))
        .def("NumStates", &PyClass::NumStates)
        .def("NumArcs", &PyClass::NumArcs, py::arg("s"))
        .def("NumInputEpsilons", &PyClass::NumInputEpsilons, py::arg("s"))
        .def("NumOutputEpsilons", &PyClass::NumOutputEpsilons, py::arg("s"))
        .def("Properties", &PyClass::Properties, py::arg("mask"),
             py::arg("test"))
        .def("Type", &PyClass::Type, "FST typename",
             py::return_value_policy::reference)
        .def("Copy", &PyClass::Copy,
             "Get a copy of this VectorFst. See Fst<>::Copy() for further "
             "doc.",
             py::arg("safe") = false, py::return_value_policy::take_ownership)
        .def_static("Read",
                    // clang-format off
            overload_cast_<std::istream&, const fst::FstReadOptions&>()(&PyClass::Read),
                    // clang-format on
                    "Reads a VectorFst from an input stream, returning nullptr "
                    "on error.",
                    py::arg("strm"), py::arg("opts"),
                    py::return_value_policy::take_ownership)
        .def_static(
            "Read", overload_cast_<const fst::string&>()(&PyClass::Read),
            "Read a VectorFst from a file, returning nullptr on error; "
            "empty "
            "filename reads from standard input.",
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
        .def_static("WriteFst", &PyClass::WriteFst<fst::StdVectorFst>,
                    py::arg("fst"), py::arg("strm"), py::arg("opts"))
        .def("InitStateIterator", &PyClass::InitStateIterator,
             "For generic state iterator construction (not normally called "
             "directly by users). Does not copy the FST.",
             py::arg("data"))
        .def("InitArcIterator", &PyClass::InitArcIterator,
             "For generic arc iterator construction (not normally called "
             "directly by users). Does not copy the FST.",
             py::arg("s"), py::arg("data"))
        .def("info",
             [](const PyClass& vector_fst) -> std::string {
               std::ostringstream os;
               auto _fst = fst::script::FstClass(vector_fst);
               auto fst_info = fst::FstInfo(*_fst.GetFst<Arc>(), true);
               PrintFstInfoImpl(fst_info);
               return os.str();
             })
        .def("__str__",
             [](const PyClass& vector_fst) -> std::string {
               std::ostringstream os;
               auto _fst = fst::script::FstClass(vector_fst);
               fst::FstPrinter<Arc>(
                   *_fst.GetFst<Arc>(), _fst.InputSymbols(),
                   _fst.OutputSymbols(),
                   nullptr,   // state symbol table, ssyms
                   false,     // false means not in acceptor format
                   false,     // false means not to show weight one
                   "      ",  // fst field separator, 6 spaces
                   ""         // missing symbol
                   )
                   .Print(&os, "standard output");
               return os.str();
             })
        .def("ToString",
             [](const PyClass& vector_fst, bool is_acceptor = false,
                bool show_weight_one = false,
                const std::string& fst_field_separator = "      ",
                const std::string& missing_symbol = "",
                const std::string& dest = "stardard output") {
               std::ostringstream os;
               auto _fst = fst::script::FstClass(vector_fst);
               fst::FstPrinter<Arc>(*_fst.GetFst<Arc>(), _fst.InputSymbols(),
                                    _fst.OutputSymbols(), nullptr, is_acceptor,
                                    show_weight_one, fst_field_separator,
                                    missing_symbol)
                   .Print(&os, dest);
               return os.str();
             },
             "see fstprint for help, e.g., fstprint --help",
             py::arg("is_acceptor") = false, py::arg("show_weight_one") = false,
             py::arg("fst_field_separator") = "      ",
             py::arg("missing_symbol") = "",
             py::arg("dest") = "stardard output");
  }
  {
    using PyClass = fst::StateIterator<fst::StdVectorFst>;
    py::class_<PyClass>(m, "StdVectorFstStateIterator")
        .def(py::init<const fst::StdVectorFst&>(), py::arg("fst"))
        .def("Done", &PyClass::Done)
        .def("Value", &PyClass::Value)
        .def("Next", &PyClass::Next)
        .def("Reset", &PyClass::Reset);
  }

  {
    using PyClass = fst::ArcIterator<fst::StdVectorFst>;
    using StateId = PyClass::StateId;
    py::class_<PyClass>(m, "StdVectorFstArcIterator")
        .def(py::init<const fst::StdVectorFst&, StateId>(), py::arg("fst"),
             py::arg("s"))
        .def("Done", &PyClass::Done)
        .def("Value", &PyClass::Value, py::return_value_policy::reference)
        .def("Next", &PyClass::Next)
        .def("Reset", &PyClass::Reset)
        .def("Seek", &PyClass::Seek, py::arg("a"))
        .def("Position", &PyClass::Position)
        .def("Flags", &PyClass::Flags)
        .def("SetFlags", &PyClass::SetFlags);
  }

  {
    using PyClass = fst::MutableArcIterator<fst::StdVectorFst>;
    using StateId = PyClass::StateId;
    py::class_<PyClass>(m, "StdVectorFstMutableArcIterator")
        .def(py::init<fst::StdVectorFst*, StateId>(), py::arg("fst"),
             py::arg("s"))
        .def("Done", &PyClass::Done)
        .def("Value", &PyClass::Value, py::return_value_policy::reference)
        .def("SetValue", &PyClass::SetValue, py::arg("arc"))
        .def("Next", &PyClass::Next)
        .def("Reset", &PyClass::Reset)
        .def("Seek", &PyClass::Seek, py::arg("a"))
        .def("Position", &PyClass::Position)
        .def("Flags", &PyClass::Flags)
        .def("SetFlags", &PyClass::SetFlags);
  }
}

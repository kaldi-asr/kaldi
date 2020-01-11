// pybind/fst/vector_fst_pybind.h

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

#ifndef KALDI_PYBIND_FST_VECTOR_FST_PYBIND_H_
#define KALDI_PYBIND_FST_VECTOR_FST_PYBIND_H_

#include "pybind/kaldi_pybind.h"

#include "fst/script/fst-class.h"
#include "fst/script/info-impl.h"
#include "fst/script/print-impl.h"
#include "fst/vector-fst.h"

void PrintFstInfoImpl(const fst::FstInfo& fstinfo, std::ostream& ostrm);

template <typename FST>
void pybind_mutable_arc_iterator_impl(py::module& m,
                                      const std::string& class_name,
                                      const std::string& class_help_doc = "") {
  using PyClass = fst::MutableArcIterator<FST>;
  using StateId = typename PyClass::StateId;

  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<FST*, StateId>(), py::arg("fst"), py::arg("s"))
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

template <typename A>
void pybind_vector_fst_impl(py::module& m, const std::string& class_name,
                            const std::string& class_help_doc = "") {
  using PyClass = fst::VectorFst<A>;
  using Arc = typename PyClass::Arc;
  using StateId = typename PyClass::StateId;
  using State = typename PyClass::State;

  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const fst::Fst<Arc>&>(), py::arg("fst"))
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
      .def("Properties", &PyClass::Properties, py::arg("mask"), py::arg("test"))
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
      .def_static("Read", overload_cast_<const fst::string&>()(&PyClass::Read),
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
      .def_static("WriteFst", &PyClass::template WriteFst<PyClass>,
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
             fst::FstPrinter<Arc>(*_fst.GetFst<Arc>(), _fst.InputSymbols(),
                                  _fst.OutputSymbols(),
                                  nullptr,  // state symbol table, ssyms
                                  false,  // false means not in acceptor format
                                  false,  // false means not to show weight one
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
           py::arg("missing_symbol") = "", py::arg("dest") = "stardard output");
}

void pybind_vector_fst(py::module& m);

#endif  // KALDI_PYBIND_FST_VECTOR_FST_PYBIND_H_

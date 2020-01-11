// pybind/fst/fst_pybind.h

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

#ifndef KALDI_PYBIND_FST_FST_PYBIND_H_
#define KALDI_PYBIND_FST_FST_PYBIND_H_

#include "pybind/kaldi_pybind.h"

#include "fst/fst.h"

template <typename A>
void pybind_state_iterator_base_impl(py::module& m,
                                     const std::string& class_name,
                                     const std::string& class_help_doc = "") {
  using PyClass = fst::StateIteratorBase<A>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def("Done", &PyClass::Done, "End of iterator?")
      .def("Value", &PyClass::Value, "Returns current state (when !Done()).")
      .def("Next", &PyClass::Next, "Advances to next state (when !Done()).")
      .def("Reset", &PyClass::Reset, "Resets to initial condition.");
}

template <typename A>
void pybind_state_iterator_data_impl(py::module& m,
                                     const std::string& class_name,
                                     const std::string& class_help_doc = "") {
  using PyClass = fst::StateIteratorData<A>;
  py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
      m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def_readwrite("base", &PyClass::base,
                     "Specialized iterator if non-zero.")
      .def_readwrite("nstates", &PyClass::nstates,
                     "Otherwise, the total number of states.");
}

template <typename A>
void pybind_arc_iterator_base_impl(py::module& m, const std::string& class_name,
                                   const std::string& class_help_doc = "") {
  using PyClass = fst::ArcIteratorBase<A>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
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

template <typename A>
void pybind_arc_iterator_data_impl(py::module& m, const std::string& class_name,
                                   const std::string& class_help_doc = "") {
  using PyClass = fst::ArcIteratorData<A>;
  py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
      m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def_readwrite("base", &PyClass::base,
                     "Specialized iterator if non-zero.")
      .def_readwrite("arcs", &PyClass::arcs, "Otherwise arcs pointer")
      .def_readwrite("narcs", &PyClass::narcs, "arc count")
      .def_readwrite("ref_count", &PyClass::ref_count,
                     "reference count if non-zero.");
}

template <typename FST>
void pybind_state_iterator_impl(py::module& m, const std::string& class_name,
                                const std::string& class_help_doc = "") {
  using PyClass = fst::StateIterator<FST>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const FST&>(), py::arg("fst"))
      .def("Done", &PyClass::Done)
      .def("Value", &PyClass::Value)
      .def("Next", &PyClass::Next)
      .def("Reset", &PyClass::Reset);
}

template <typename FST>
void pybind_arc_iterator_impl(py::module& m, const std::string& class_name,
                              const std::string& class_help_doc = "") {
  using PyClass = fst::ArcIterator<FST>;
  using StateId = typename FST::StateId;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const FST&, StateId>(), py::arg("fst"), py::arg("s"))
      .def("Done", &PyClass::Done)
      .def("Value", &PyClass::Value, py::return_value_policy::reference)
      .def("Next", &PyClass::Next)
      .def("Reset", &PyClass::Reset)
      .def("Seek", &PyClass::Seek, py::arg("a"))
      .def("Position", &PyClass::Position)
      .def("Flags", &PyClass::Flags)
      .def("SetFlags", &PyClass::SetFlags);
}

template <typename A>
void pybind_fst_impl(py::module& m, const std::string& class_name,
                     const std::string& class_help_doc = "") {
  using PyClass = fst::Fst<A>;
  using Arc = typename PyClass::Arc;
  using StateId = typename PyClass::StateId;
  using Weight = typename PyClass::Weight;

  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
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
      .def("Copy", &PyClass::Copy,
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
           py::arg("s"), py::arg("data"));
}

void pybind_fst(py::module& m);

#endif  // KALDI_PYBIND_FST_FST_PYBIND_H_

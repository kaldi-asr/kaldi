// pybind/fst/arc_pybind.h

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

#ifndef KALDI_PYBIND_FST_ARC_PYBIND_H_
#define KALDI_PYBIND_FST_ARC_PYBIND_H_

#include "pybind/kaldi_pybind.h"

#include "fst/arc.h"

template <typename W>
void pybind_arc_impl(py::module& m, const std::string& class_name,
                     const std::string& class_help_doc = "") {
  using PyClass = fst::ArcTpl<W>;
  using Weight = typename PyClass::Weight;
  using Label = typename PyClass::Label;
  using StateId = typename PyClass::StateId;

  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<Label, Label, Weight, StateId>(), py::arg("ilabel"),
           py::arg("olabel"), py::arg("weight"), py::arg("nextstate"))
      .def(py::init<const PyClass&>(), py::arg("weight"))
      .def_readwrite("ilabel", &PyClass::ilabel)
      .def_readwrite("olabel", &PyClass::olabel)
      .def_readwrite("weight", &PyClass::weight)
      .def_readwrite("nextstate", &PyClass::nextstate)
      .def("__str__",
           [](const PyClass& arc) {
             std::ostringstream os;
             os << "(ilabel: " << arc.ilabel << ", "
                << "olabel: " << arc.olabel << ", "
                << "weight: " << arc.weight << ", "
                << "nextstate: " << arc.nextstate << ")";
             return os.str();
           })
      .def_static("Type", &PyClass::Type, py::return_value_policy::reference);
}

void pybind_arc(py::module& m);

#endif  // KALDI_PYBIND_FST_ARC_PYBIND_H_

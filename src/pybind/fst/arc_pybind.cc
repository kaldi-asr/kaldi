// pybind/fst/arc_pybind.cc

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

#include "fst/arc_pybind.h"

#include "fst/arc.h"

void pybind_arc(py::module& m) {
  {
    using PyClass = fst::StdArc;
    using Weight = PyClass::Weight;
    using Label = int;
    using StateId = int;

    py::class_<PyClass>(m, "StdArc")
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
                  << "weight: " << arc.weight.Value() << ", "
                  << "nextstate: " << arc.nextstate << ")";
               return os.str();
             })
        .def_static("Type", &PyClass::Type);
  }
}

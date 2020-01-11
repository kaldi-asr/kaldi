// pybind/fst/weight_pybind.cc

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

#include "fst/weight_pybind.h"

#include "fst/float-weight.h"

void pybind_weight(py::module& m) {
  {
    using PyClass = fst::FloatWeight;
    py::class_<PyClass>(m, "FloatWeight")
        .def(py::init<>())
        .def(py::init<float>(), py::arg("f"))
        .def(py::init<const PyClass&>(), py::arg("weight"))
        .def("Value", &PyClass::Value, py::return_value_policy::reference)
        .def("Hash", &PyClass::Hash)
        .def("__eq__",
             [](const PyClass& w1, const PyClass& w2) { return w1 == w2; })
        .def("__str__", [](const PyClass& w) {
          std::ostringstream os;
          os << w.Value();
          return os.str();
        });
  }
  {
    using PyClass = fst::TropicalWeight;
    py::class_<PyClass, fst::FloatWeight>(m, "TropicalWeight")
        .def(py::init<>())
        .def(py::init<float>(), py::arg("f"))
        .def(py::init<const PyClass&>(), py::arg("weight"))
        .def("Member", &PyClass::Member)
        .def("Quantize", &PyClass::Quantize, py::arg("delta") = fst::kDelta)
        .def("Reverse", &PyClass::Reverse)
        .def_static("Zero", &PyClass::Zero)
        .def_static("One", &PyClass::One)
        .def_static("NoWeight", &PyClass::NoWeight)
        .def_static("Type", &PyClass::Type, py::return_value_policy::reference)
        .def_static("Properties", &PyClass::Properties);

    m.def("Plus", [](const PyClass& w1, const PyClass& w2) {
      return fst::Plus(w1, w2);
    });

    m.def("Times", [](const PyClass& w1, const PyClass& w2) {
      return fst::Times(w1, w2);
    });

    m.def("Divide", [](const PyClass& w1, const PyClass& w2) {
      return fst::Divide(w1, w2);
    });
  }
}

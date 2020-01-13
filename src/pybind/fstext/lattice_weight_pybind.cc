// pybind/fstext/lattice_weight_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China
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

#include "fstext/lattice_weight_pybind.h"

#include "fstext/lattice-weight.h"

namespace {

template <typename FloatType>
void pybind_lattice_weight_impl(py::module& m, const std::string& class_name,
                                const std::string& class_help_doc = "") {
  using PyClass = fst::LatticeWeightTpl<FloatType>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<FloatType, FloatType>(), py::arg("a"), py::arg("b"))
      .def(py::init<const PyClass&>(), py::arg("other"))
      .def("Value1", &PyClass::Value1)
      .def("Value2", &PyClass::Value2)
      .def("SetValue1", &PyClass::SetValue1, py::arg("f"))
      .def("SetValue2", &PyClass::SetValue2, py::arg("f"))
      .def("Reverse", &PyClass::Reverse)
      .def_static("Zero", &PyClass::Zero)
      .def_static("One", &PyClass::One)
      .def_static("Type", &PyClass::Type)
      .def_static("NoWeight", &PyClass::NoWeight)
      .def("Member", &PyClass::Member)
      .def("Quantize", &PyClass::Quantize, py::arg("delta") = fst::kDelta)
      .def("Properties", &PyClass::Properties)
      .def("Hash", &PyClass::Hash)
      .def("__eq__",
           [](const PyClass& wa, const PyClass& wb) { return wa == wb; })
      .def("__ne__",
           [](const PyClass& wa, const PyClass& wb) { return wa != wb; })
      .def("__str__", [](const PyClass& lat_weight) {
        std::ostringstream os;
        os << "Value1 (lm cost): " << lat_weight.Value1() << "\n";
        os << "Value2 (acoustic cost): " << lat_weight.Value2() << "\n";
        return os.str();
      });

  m.def(
      "ScaleTupleWeight",
      (PyClass(*)(const PyClass&, const std::vector<std::vector<FloatType>>&))(
          &fst::ScaleTupleWeight<FloatType, FloatType>),
      "ScaleTupleWeight is a function defined for LatticeWeightTpl and "
      "CompactLatticeWeightTpl that mutliplies the pair (value1_, value2_) "
      "by a 2x2 matrix.  Used, for example, in applying acoustic scaling.",
      py::arg("w"), py::arg("scale"));
}

template <typename WeightType, typename IntType>
void pybind_compact_lattice_weight_impl(
    py::module& m, const std::string& class_name,
    const std::string& class_help_doc = "") {
  using PyClass = fst::CompactLatticeWeightTpl<WeightType, IntType>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def(py::init<const WeightType&, const std::vector<IntType>&>(),
           py::arg("w"), py::arg("s"))
      .def("Weight", &PyClass::Weight, py::return_value_policy::reference)
      .def("String", &PyClass::String, py::return_value_policy::reference)
      .def("SetWeight", &PyClass::SetWeight, py::arg("w"))
      .def("SetString", &PyClass::SetString, py::arg("s"))
      .def_static("Zero", &PyClass::Zero)
      .def_static("One", &PyClass::One)
      .def_static("GetIntSizeString", &PyClass::GetIntSizeString)
      .def_static("Type", &PyClass::Type)
      .def_static("NoWeight", &PyClass::NoWeight)
      .def("Reverse", &PyClass::Reverse)
      .def("Member", &PyClass::Member)
      .def("Quantize", &PyClass::Quantize, py::arg("delta") = fst::kDelta)
      .def("Properties", &PyClass::Properties)
      .def("__eq__",
           [](const PyClass& w1, const PyClass& w2) { return w1 == w2; })
      .def("__ne__",
           [](const PyClass& w1, const PyClass& w2) { return w1 != w2; })
      .def("__str__", [](const PyClass& lat_weight) {
        std::ostringstream os;
        os << lat_weight;
        return os.str();
      });
}

}  // namespace

void pybind_lattice_weight(py::module& m) {
  pybind_lattice_weight_impl<float>(m, "LatticeWeight",
                                    "Contain two values: value1 is the lm cost "
                                    "and value2 is the acoustic cost.");
  pybind_compact_lattice_weight_impl<fst::LatticeWeightTpl<float>, int>(
      m, "CompactLatticeWeight",
      "Contain two members: fst::LatticeWeight and std::vector<int>");
}

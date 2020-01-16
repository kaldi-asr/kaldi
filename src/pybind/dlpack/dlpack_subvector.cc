// pybind/dlpack/dlpack_subvector.cc

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

#include "dlpack/dlpack_subvector.h"

#include "dlpack/dlpack_pybind.h"

namespace kaldi {

template class _DLPackSubVector<float, float>;
template class _DLPackSubVector<int, int>;
template class DLPackCuSubVector<float>;

}  // namespace kaldi

using namespace kaldi;

namespace {

template <typename SubVectorType>
void pybind_DL_subvector_impl(py::module& m, const std::string& class_name,
                              const std::string& class_doc = "") {
  using PyClass = SubVectorType;
  using Type = typename PyClass::type;
  py::class_<PyClass>(m, class_name.c_str(), class_doc.c_str())
      .def("Dim", &PyClass::Dim, "Returns the dimension of the vector.")
      .def("__repr__",
           [](const PyClass& v) -> std::string {
             std::ostringstream os;
             std::string sep = "";

             os << "[";
             for (auto i : v) {
               os << sep << i;
               sep = ", ";
             }
             os << "]";
             return os.str();
           })
      .def("__getitem__", [](const PyClass& v, int i) { return v[i]; })
      .def("__setitem__", [](PyClass& v, int i, Type val) { v[i] = val; })
      .def("numpy",
           [](py::object obj) {
             auto* v = obj.cast<PyClass*>();
             return py::array_t<Type>({v->Dim()},      // shape
                                      {sizeof(Type)},  // stride in bytes
                                      v->Data(),       // ptr
                                      obj); /* it will increase the reference
                                               count of **this** vector */
           })
      .def("from_dlpack",
           [](py::capsule* capsule) {
             return SubVectorFromDLPack<Type>(capsule);
           },
           py::return_value_policy::take_ownership);
}

}  // namespace

void pybind_DL_subvector(py::module& m) {
  // Note that the float type sub vectors
  // are wrapped in kaldi_vector_pybind.cc
  pybind_DL_subvector_impl<DLPackSubVector<int>>(m, "DLPackIntSubVector",
                                                 "int32 DLPack subvector");
}

// pybind/util/kaldi_holder_inl_pybind.h

// Copyright 2020   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../COPYING for clarification regarding multiple authors
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

#ifndef KALDI_PYBIND_UTIL_KALDI_HOLDER_INL_PYBIND_H_
#define KALDI_PYBIND_UTIL_KALDI_HOLDER_INL_PYBIND_H_

#include "pybind/kaldi_pybind.h"

#include "util/kaldi-holder-inl.h"

using namespace kaldi;

template <class BasicType>
void pybind_basic_vector_holder(py::module& m, const std::string& class_name,
                                const std::string& class_help_doc = "") {
  using PyClass = BasicVectorHolder<BasicType>;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<>())
      .def("Clear", &PyClass::Clear)
      .def("Read", &PyClass::Read, py::arg("is"))
      .def_static("IsReadInBinary", &PyClass::IsReadInBinary)
      .def("Value", &PyClass::Value, py::return_value_policy::reference);
  // TODO(fangjun): wrap other methods when needed
}

#endif  // KALDI_PYBIND_UTIL_KALDI_HOLDER_INL_PYBIND_H_

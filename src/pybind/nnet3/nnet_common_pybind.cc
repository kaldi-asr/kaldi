// pybind/nnet3/nnet_common_pybind.cc

// Copyright 2019   Mobvoi AI Lab, Beijing, China
//                  (author: Fangjun Kuang, Yaguang Hu, Jian Wang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/nnet_common_pybind.h"

#include "nnet3/nnet-common.h"

using namespace kaldi;
using namespace kaldi::nnet3;

void pybind_nnet_common(py::module& m) {
  {
    // Index is need by NnetChainSupervision in nnet_chain_example_pybind.cc
    using PyClass = Index;
    py::class_<PyClass>(
        m, "Index",
        "struct Index is intended to represent the various indexes by which we "
        "number the rows of the matrices that the Components process: mainly "
        "'n', the index of the member of the minibatch, 't', used for the "
        "frame index in speech recognition, and 'x', which is a catch-all "
        "extra index which we might use in convolutional setups or for other "
        "reasons.  It is possible to extend this by adding new indexes if "
        "needed.")
        .def(py::init<>())
        .def(py::init<int, int, int>(), py::arg("n"), py::arg("t"),
             py::arg("x") = 0)
        .def_readwrite("n", &PyClass::n, "member-index of minibatch, or zero.")
        .def_readwrite("t", &PyClass::t, "time-frame.")
        .def_readwrite("x", &PyClass::x,
                       "this may come in useful in convolutional approaches. "
                       "it is possible to add extra index here, if needed.")
        .def("__eq__",
             [](const PyClass& a, const PyClass& b) { return a == b; })
        .def("__ne__",
             [](const PyClass& a, const PyClass& b) { return a != b; })
        .def("__lt__", [](const PyClass& a, const PyClass& b) { return a < b; })
        .def(py::self + py::self)
        .def(py::self += py::self)
        // TODO(fangjun): other methods can be wrapped when needed
        ;
  }
}

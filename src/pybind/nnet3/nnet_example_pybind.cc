// pybind/nnet3/nnet_example_pybind.cc

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

#include "nnet3/nnet_example_pybind.h"

#include "nnet3/nnet-example.h"

using namespace kaldi;
using namespace kaldi::nnet3;

void pybind_nnet_example(py::module& m) {
  {
    using PyClass = NnetIo;
    py::class_<PyClass>(m, "NnetIo")
        .def(py::init<>())
        .def_readwrite("name", &PyClass::name,
                       "the name of the input in the neural net; in simple "
                       "setups it will just be 'input'.");
    // TODO(fangjun): other constructors, fields and methods can be wrapped when
  }
}

// pybind/tests/test_dlpack_subvector.cc

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

#include "dlpack/dlpack_pybind.h"

// TODO(fangjun): remove this file if neccessary
// this file is to show that we can pass
// a DLPackSubVector to a function in C++
// that accepts a pointer/reference to VectorBase<float>

using namespace kaldi;
namespace {
void Hello(VectorBase<float>& vref, VectorBase<float>* vptr) {
  KALDI_LOG << "dim of vref is: " << vref.Dim();
  KALDI_LOG << "vptr[0] = " << (*vptr)(0);
}
}

void test_dlpack(py::module& m) {
  m.def("Hello", &Hello,
        "For test only. Pass a DLPackSubVector to C++ that accepts a pointer "
        "to VectorBase<float>");
}

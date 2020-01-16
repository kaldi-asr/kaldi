// pybind/dlpck/dlpack_pybind.h

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

#ifndef KALDI_PYBIND_DLPACK_DLPACK_PYBIND_H_
#define KALDI_PYBIND_DLPACK_DLPACK_PYBIND_H_

#include "pybind/kaldi_pybind.h"

#include "dlpack/dlpack_submatrix.h"
#include "dlpack/dlpack_subvector.h"

void pybind_dlpack(py::module& m);

namespace kaldi {

// Inside the function, we will use
// ```Vector<float>* v = obj->cast<Vector<float>*>();```
// if it fails, it will throw.
py::capsule VectorToDLPack(py::object* obj);
py::capsule MatrixToDLPack(py::object* obj);
py::capsule CuVectorToDLPack(py::object* obj);
py::capsule CuMatrixToDLPack(py::object* obj);

template <typename DType>
DLPackSubVector<DType>* SubVectorFromDLPack(py::capsule* capsule);

DLPackSubMatrix<float>* SubMatrixFromDLPack(py::capsule* capsule);
DLPackCuSubVector<float>* CuSubVectorFromDLPack(py::capsule* capsule);
DLPackCuSubMatrix<float>* CuSubMatrixFromDLPack(py::capsule* capsule);

}  // namespace kaldi

#endif  // KALDI_PYBIND_DLPACK_DLPACK_PYBIND_H_

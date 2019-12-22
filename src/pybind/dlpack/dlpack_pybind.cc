// pybind/dlpack/dlpack_pybind.cc

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

#include "dlpack/dlpack.h"

#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"

using namespace kaldi;

void pybind_dlpack(py::module& m) {
  m.def("ToSubVector", [](py::capsule* capsule) {
    DLManagedTensor* managed_tensor = *capsule;
    // (fangjun): the above assignment will either throw or succeed with a
    // non-null ptr so no need to check for nullptr below

    auto* tensor = &managed_tensor->dl_tensor;

    // we support only 1-D tensor
    KALDI_ASSERT(tensor->ndim == 1);

    // we support only float (single precision, 32-bit) tensor
    KALDI_ASSERT(tensor->dtype.code == kDLFloat);
    KALDI_ASSERT(tensor->dtype.bits == 32);
    KALDI_ASSERT(tensor->dtype.lanes == 1);

    return SubVector<float>(reinterpret_cast<float*>(tensor->data),
                            tensor->shape[0]);
  });

  m.def("ToSubMatrix", [](py::capsule* capsule) {
    DLManagedTensor* managed_tensor = *capsule;

    auto* tensor = &managed_tensor->dl_tensor;

    // we support only 2-D tensor
    KALDI_ASSERT(tensor->ndim == 2);

    // we support only float (single precision, 32-bit) tensor
    KALDI_ASSERT(tensor->dtype.code == kDLFloat);
    KALDI_ASSERT(tensor->dtype.bits == 32);
    KALDI_ASSERT(tensor->dtype.lanes == 1);

    // DLPack assumes row major, so we use strides[0]
    return SubMatrix<float>(reinterpret_cast<float*>(tensor->data),
                            tensor->shape[0], tensor->shape[1],
                            tensor->strides[0]);
  });

  m.def("ToCuSubVector", [](py::capsule* capsule) {
#if HAVE_CUDA == 1
    DLManagedTensor* managed_tensor = *capsule;

    auto* tensor = &managed_tensor->dl_tensor;

    // we support only 1-D tensor
    KALDI_ASSERT(tensor->ndim == 1);

    // we support only float (single precision, 32-bit) tensor
    KALDI_ASSERT(tensor->dtype.code == kDLFloat);
    KALDI_ASSERT(tensor->dtype.bits == 32);
    KALDI_ASSERT(tensor->dtype.lanes == 1);

    return CuSubVector<float>(reinterpret_cast<float*>(tensor->data),
                              tensor->shape[0]);
#else
      KALDI_ERR << "Kaldi is not compiled with GPU!"
#endif
  });

  m.def("ToCuSubMatrix", [](py::capsule* capsule) {
#if HAVE_CUDA == 1
    DLManagedTensor* managed_tensor = *capsule;

    auto* tensor = &managed_tensor->dl_tensor;

    // we support only 2-D tensor
    KALDI_ASSERT(tensor->ndim == 2);

    // we support only float (single precision, 32-bit) tensor
    KALDI_ASSERT(tensor->dtype.code == kDLFloat);
    KALDI_ASSERT(tensor->dtype.bits == 32);
    KALDI_ASSERT(tensor->dtype.lanes == 1);

    // DLPack assumes row major, so we use strides[0]
    return CuSubMatrix<float>(reinterpret_cast<float*>(tensor->data),
                              tensor->shape[0], tensor->shape[1],
                              tensor->strides[0]);
#else
      KALDI_ERR << "Kaldi is not compiled with GPU!"
#endif
  });
}

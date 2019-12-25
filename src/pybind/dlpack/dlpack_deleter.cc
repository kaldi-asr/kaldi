// pybind/dlpack/dlpack_deleter.cc

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

#include "dlpack/dlpack_deleter.h"

#include <Python.h>

namespace kaldi {

void DLManagedTensorDeleter(struct DLManagedTensor* dl_managed_tensor) {
  // data is shared, so do NOT free data

  // `shape` is created with `new[]`
  delete[] dl_managed_tensor->dl_tensor.shape;

  // `strides` is created with `new[]`
  delete[] dl_managed_tensor->dl_tensor.strides;

  if (dl_managed_tensor->manager_ctx) {
    PyObject* obj = reinterpret_cast<PyObject*>(dl_managed_tensor->manager_ctx);
    Py_XDECREF(obj);
  }

  // now delete the `DLManagedTensor` which is created with `new`
  delete dl_managed_tensor;
}

}  // namespace kaldi

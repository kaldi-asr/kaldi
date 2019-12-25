// pybind/dlpck/dlpack_subvector.h

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

#ifndef KALDI_PYBIND_DLPACK_DLPACK_SUBVECTOR_H_
#define KALDI_PYBIND_DLPACK_DLPACK_SUBVECTOR_H_

#include "dlpack/dlpack.h"

#include "cudamatrix/cu-vector.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {

/*
The following comment is copied from
https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L387

```
DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data,
"dltensor");
// atensor steals the ownership of the underlying storage. It also passes a
// destructor function that will be called when the underlying storage goes
// out of scope. When the destructor is called, the dlMTensor is destructed too.
auto atensor = at::fromDLPack(dlMTensor);
```

We create `DLPackSubVector` to free `DLManagedTensor` passed
from Python by `PyCapsule`.
*/

template <typename Real>
class DLPackSubVector : public SubVector<Real> {
 public:
  // Note that `const Real* data` will be `const_cast`
  // to `Real *`
  DLPackSubVector(const Real* data, MatrixIndexT length, DLManagedTensor* ptr)
      : SubVector<Real>(data, length), dl_managed_tensor_(ptr) {}

  DLPackSubVector(const DLPackSubVector&) = delete;
  DLPackSubVector& operator=(const DLPackSubVector&) = delete;

  ~DLPackSubVector() {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
    }
  }

 private:
  DLManagedTensor* dl_managed_tensor_ = nullptr;
};

template <typename Real>
class DLPackCuSubVector : public CuSubVector<Real> {
 public:
  DLPackCuSubVector(const Real* data, MatrixIndexT length, DLManagedTensor* ptr)
      : CuSubVector<Real>(data, length), dl_managed_tensor_(ptr) {}

  DLPackCuSubVector(const DLPackCuSubVector&) = delete;
  DLPackCuSubVector& operator=(const DLPackCuSubVector&) = delete;

  ~DLPackCuSubVector() {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
    }
  }

 private:
  DLManagedTensor* dl_managed_tensor_ = nullptr;
};

}  // namespace kaldi

#endif  // KALDI_PYBIND_DLPACK_DLPACK_SUBVECTOR_H_

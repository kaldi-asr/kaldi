// pybind/dlpck/dlpack_submatrix.h

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

#ifndef KALDI_PYBIND_DLPACK_DLPACK_SUBMATRIX_H_
#define KALDI_PYBIND_DLPACK_DLPACK_SUBMATRIX_H_

#include "dlpack/dlpack.h"

#include "cudamatrix/cu-matrix.h"
#include "matrix/kaldi-matrix.h"

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

We cannot construct a `SuMatrix` from dlpack since `SubMatrix`
cannot invoke the `deleter` of `DLManagedTensor` in the destructor.

Therefore, we create `DLPackSubMatrix` to free the `DLManagedTensor` passed
from Python by `PyCapsule`.

`DLPackSubMatrix` is a subclass of `SubMatrix` and we can pass a pointer
of `DLPackSubMatrix` to any function that accepts `SubMatrix<float>*`
or `MatrixBase<float>*`.
*/

template <typename Real>
class DLPackSubMatrix : public SubMatrix<Real> {
 public:
  // note that unlike with `DLPackSubVector`, it requires
  // `Real* data` instead of `const Real* data`.
  DLPackSubMatrix(Real* data, MatrixIndexT num_rows, MatrixIndexT num_cols,
                  MatrixIndexT stride, DLManagedTensor* ptr)
      : SubMatrix<Real>(data, num_rows, num_cols, stride),
        dl_managed_tensor_(ptr) {}

  DLPackSubMatrix(const DLPackSubMatrix&) = delete;
  DLPackSubMatrix& operator=(const DLPackSubMatrix&) = delete;

  ~DLPackSubMatrix() {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
    }
    // TODO(fangjun): remove this log
    std::cout << "dlpack SubMatrix is called" << std::endl;
  }

 private:
  DLManagedTensor* dl_managed_tensor_ = nullptr;
};

template <typename Real>
class DLPackCuSubMatrix : public CuSubMatrix<Real> {
 public:
  DLPackCuSubMatrix(const Real* data, const MatrixIndexT num_rows,
                    const MatrixIndexT num_cols, const MatrixIndexT stride,
                    DLManagedTensor* ptr)
      : CuSubMatrix<Real>(data, num_rows, num_cols, stride),
        dl_managed_tensor_(ptr) {}

  DLPackCuSubMatrix(const DLPackCuSubMatrix&) = delete;
  DLPackCuSubMatrix& operator=(const DLPackCuSubMatrix&) = delete;

  ~DLPackCuSubMatrix() {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
    }
    std::cout << "dlpack CuSubMatrix is called" << std::endl;
  }

 private:
  DLManagedTensor* dl_managed_tensor_ = nullptr;
};

}  // namespace kaldi

#endif  // KALDI_PYBIND_DLPACK_DLPACK_SUBMATRIX_H_

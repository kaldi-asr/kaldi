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

#include <string.h>  // for strcmp

#include "dlpack/dlpack.h"
#include "dlpack/dlpack_deleter.h"

#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"

namespace {

using namespace kaldi;

// refer to
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L375
// https://github.com/microsoft/onnxruntime-tvm/blob/master/python/tvm/_ffi/_ctypes/ndarray.py#L28
// https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx#L66
// PyTorch, TVM and CuPy name the created dltensor to be `dltensor`
const char* kDLPackTensorName = "dltensor";

// refer to
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L402
// https://github.com/apache/incubator-tvm/blob/master/python/tvm/_ffi/_ctypes/ndarray.py#L29
// https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx#L62
// PyTorch, TVM and CuPy name the used dltensor to be `used_dltensor`
const char* kDLPackUsedTensorName = "used_dltensor";

DLManagedTensor* CreateDLManagedtensor(DLDeviceType device_type, int device_id,
                                       void* data) {
  // As SubVector/SubMatrix/CuSubVector/CuSumMatrix
  // all require a DLManagedTensor, we put the shared
  // code here to avoid duplicates

  // the created `managed_tensor` will be freed in
  // `DLManagedTensorDeleter`, which does not free `data`,
  // so no memory leak here
  auto* managed_tensor = new DLManagedTensor();
  managed_tensor->manager_ctx = nullptr;

  // setup the deleter to free allocated memory.
  // refer to
  // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L361
  // for how and when the deleter is invoked.
  managed_tensor->deleter = &DLManagedTensorDeleter;

  auto* tensor = &managed_tensor->dl_tensor;
  tensor->data = data;
  tensor->ctx.device_type = device_type;
  tensor->ctx.device_id = device_id;

  tensor->dtype.code = kDLFloat;
  tensor->dtype.bits = 32;  // single precision float
  tensor->dtype.lanes = 1;

  tensor->byte_offset = 0;

  // ndim, shape, strides are set outside
  return managed_tensor;
}

DLManagedTensor* ConsumeDLManagedtensor(py::capsule* capsule,
                                        DLDeviceType device_type, int device_id,
                                        int ndim) {
  // check the name of the capsule
  if (strcmp(kDLPackTensorName, capsule->name()) != 0) {
    // the following error message is modified from
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L384
    KALDI_ERR << "Expected capsule name: " << kDLPackTensorName << "\n"
              << "But got: " << capsule->name() << "\n"
              << "Note that DLTensor capsules can be consumed only once,\n"
              << "so you might have already constructed a tensor from it once.";
    return nullptr;
  }

  PyCapsule_SetName(capsule->ptr(), kDLPackUsedTensorName);

  DLManagedTensor* managed_tensor = *capsule;
  // (fangjun): the above assignment will either throw or succeed with a
  // non-null ptr so no need to check for nullptr below

  auto* tensor = &managed_tensor->dl_tensor;

  KALDI_ASSERT(tensor->ndim == ndim);

  // we support only float (single precision, 32-bit) tensor
  KALDI_ASSERT(tensor->dtype.code == kDLFloat);
  KALDI_ASSERT(tensor->dtype.bits == 32);
  KALDI_ASSERT(tensor->dtype.lanes == 1);

  auto* ctx = &tensor->ctx;
  KALDI_ASSERT(ctx->device_type == device_type);
  if (device_type == kDLGPU) {
    KALDI_ASSERT(ctx->device_id == device_id);
  }
  return managed_tensor;
}

// this function is modified from
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L355
void DLPackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* managed_tensor = reinterpret_cast<DLManagedTensor*>(
      PyCapsule_GetPointer(data, kDLPackTensorName));

  // if managed_tensor is nullptr, then there is an error; we MUST
  // clear the error flag manually by using PyErr_Clear().
  // See https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_GetPointer

  if (managed_tensor && managed_tensor->deleter) {
    // the managed_tensor has not been consumed, call deleter ourselves
    managed_tensor->deleter(managed_tensor);
  } else {
    // the managed_tensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
}

}  // namespace

namespace kaldi {

/*
To test the deleter function of `VectorToDLPack`, you can use the following
methods (first print a log inside the deleter function)

(1) When the capsule has not been consumed, call the deleter by ourselves

```python
import kaldi

v = kaldi.FloatVector(2)

dlpack = v.to_dlpack()
print('del dlpack')
del dlpack
print('after del dlpack')
```

You should see

```
del dlpack
<the log message you added in the deleter>
after del dlpack
```

(2) When the capsule has been consumed, the deleter is also invoked.

```python
import torch
from torch.utils.dlpack import from_dlpack

import kaldi

v = kaldi.FloatVector(2)

dlpack = v.to_dlpack()
tensor = from_dlpack(dlpack)
print('del tensor')
del tensor
print('after del tensor')
```

You should see

```
del tensor
<the log message you added in the deleter>
after del tensor
```
*/

py::capsule VectorToDLPack(VectorBase<float>* v) {
  auto* managed_tensor = CreateDLManagedtensor(kDLCPU, 0, v->Data());
  auto* tensor = &managed_tensor->dl_tensor;

  tensor->ndim = 1;

  // `shape` and `strides` are freed in `DLManagedTensorDeleter`, so
  // no memory leak here .
  tensor->shape = new int64_t[1];
  tensor->shape[0] = v->Dim();

  tensor->strides = new int64_t[1];
  tensor->strides[0] = 1;

  PyObject* capsule =
      PyCapsule_New(managed_tensor, kDLPackTensorName, DLPackCapsuleDestructor);
  bool is_borrowed = false;
  return py::object(capsule, is_borrowed);
  // `py::object` will free `capsule` created above in its destructor
  // `py::object` is implicitly converted to `py::capsule`
}

py::capsule MatrixToDLPack(MatrixBase<float>* m) {
  auto* managed_tensor = CreateDLManagedtensor(kDLCPU, 0, m->Data());
  auto* tensor = &managed_tensor->dl_tensor;

  tensor->ndim = 2;

  // `shape` and `strides` are freed in `DLManagedTensorDeleter`, so
  // no memory leak here
  tensor->shape = new int64_t[2];
  tensor->shape[0] = m->NumRows();
  tensor->shape[1] = m->NumCols();

  tensor->strides = new int64_t[2];
  tensor->strides[0] = m->Stride();
  tensor->strides[1] = 1;

  PyObject* capsule =
      PyCapsule_New(managed_tensor, kDLPackTensorName, DLPackCapsuleDestructor);
  bool is_borrowed = false;
  return py::object(capsule, is_borrowed);
}

py::capsule CuVectorToDLPack(CuVectorBase<float>* v) {
#if HAVE_CUDA == 1
  auto* managed_tensor =
      CreateDLManagedtensor(kDLGPU, CuDevice::GetCurrentDeviceId(), v->Data());

  auto* tensor = &managed_tensor->dl_tensor;

  tensor->ndim = 1;

  // `shape` and `strides` are freed in `DLManagedTensorDeleter`,
  // so no memory leak here.
  tensor->shape = new int64_t[1];
  tensor->shape[0] = v->Dim();

  tensor->strides = new int64_t[1];
  tensor->strides[0] = 1;

  PyObject* capsule =
      PyCapsule_New(managed_tensor, kDLPackTensorName, DLPackCapsuleDestructor);
  bool is_borrowed = false;
  return py::object(capsule, is_borrowed);
#else
  KALDI_ERR << "Kaldi is not compiled with GPU!";
  return py::none();
#endif
}

py::capsule CuMatrixToDLPack(CuMatrixBase<float>* m) {
#if HAVE_CUDA == 1
  auto* managed_tensor =
      CreateDLManagedtensor(kDLGPU, CuDevice::GetCurrentDeviceId(), m->Data());

  auto* tensor = &managed_tensor->dl_tensor;

  tensor->ndim = 2;

  // `shape` and `strides` are freed in `DLManagedTensorDeleter`,
  // so no memory leak here
  tensor->shape = new int64_t[2];
  tensor->shape[0] = m->NumRows();
  tensor->shape[1] = m->NumCols();

  tensor->strides = new int64_t[2];
  tensor->strides[0] = m->Stride();
  tensor->strides[1] = 1;

  PyObject* capsule =
      PyCapsule_New(managed_tensor, kDLPackTensorName, DLPackCapsuleDestructor);
  bool is_borrowed = false;
  return py::object(capsule, is_borrowed);
#else
  KALDI_ERR << "Kaldi is not compiled with GPU!";
  return py::none();
#endif
}

// As the destructor of `VectorBase<float>` is not `virtual`
// we cannot return a `VectorBase<float>*` or `SubVector<float>*`.
DLPackSubVector<float>* SubVectorFromDLPack(py::capsule* capsule) {
  auto* managed_tensor = ConsumeDLManagedtensor(capsule, kDLCPU, 0, 1);
  auto* tensor = &managed_tensor->dl_tensor;

  // we use `py::return_value_policy::take_ownership`
  // and rely on Python to delete the allocated memory.
  //
  // https://pybind11.readthedocs.io/en/stable/advanced/functions.html
  // says: "When Python’s garbage collector eventually deletes the Python
  // wrapper, pybind11 will also attempt to delete the C++ instance (via
  // operator delete()) due to the implied ownership."
  //
  // Therefore, we use `new` instead of `malloc` with `placement new` here.
  return new DLPackSubVector<float>(reinterpret_cast<float*>(tensor->data),
                                    tensor->shape[0], managed_tensor);
  // clang-format off
/*
You can use the following method to check that the above allocated
memory is indeed freed.

1. Put a `std::cout` statement or use `KALDI_LOG` inside
the destructor of `DLPackSubVector`

2. Run the following Python test code

```python
import torch
from torch.utils.dlpack import to_dlpack
import kaldi

tensor = torch.tensor([1, 2]).float()

dlpack = to_dlpack(tensor)
v = kaldi.SubVectorFromDLPack(dlpack)
print('del v')
del v
print('after del v')
```
You should see

```
del v
<the log message you added inside the destructor of DLPackSubVector>
after del v
```

Since the destructor is called, we can trust Pybind11 that it
invokes the `delete` operator to free the memory allocated by `new`.
*/
  // clang-format on
}

DLPackSubMatrix<float>* SubMatrixFromDLPack(py::capsule* capsule) {
  auto* managed_tensor = ConsumeDLManagedtensor(capsule, kDLCPU, 0, 2);
  auto* tensor = &managed_tensor->dl_tensor;

  // DLPack assumes row major, so we use strides[0]
  return new DLPackSubMatrix<float>(reinterpret_cast<float*>(tensor->data),
                                    tensor->shape[0], tensor->shape[1],
                                    tensor->strides[0], managed_tensor);
}

DLPackCuSubVector<float>* CuSubVectorFromDLPack(py::capsule* capsule) {
#if HAVE_CUDA == 1
  auto* managed_tensor = ConsumeDLManagedtensor(
      capsule, kDLGPU, CuDevice::GetCurrentDeviceId(), 1);
  auto* tensor = &managed_tensor->dl_tensor;

  return new DLPackCuSubVector<float>(reinterpret_cast<float*>(tensor->data),
                                      tensor->shape[0], managed_tensor);
#else
  KALDI_ERR << "Kaldi is not compiled with GPU!";
  return py::none();
#endif
  // clang-format off
/*
You can use the following methods to check that the
desturctor of `DLPackCuSubVector` is invoked from Python
and thus there is no memory leak in the above `new statement`.

1. print some log in the destructor of DLPackCuSubVector

2. try the following code

```python
import torch
from torch.utils.dlpack import to_dlpack
import kaldi

device_id = 0
device = torch.device('cuda', device_id)

tensor = torch.tensor([1, 2]).float()
tensor = tensor.to(device)
dlpack = to_dlpack(tensor)

kaldi.SelectGpuDevice(device_id=device_id)
v = kaldi.CuSubVectorFromDLPack(dlpack)
print('del v')
del v
print('after del v')
```

You shoulde see something like this:

```
del v
<the log message you added inside the destructor>
after del v
```
 */
  // clang-format on
}

DLPackCuSubMatrix<float>* CuSubMatrixFromDLPack(py::capsule* capsule) {
#if HAVE_CUDA == 1
  auto* managed_tensor = ConsumeDLManagedtensor(
      capsule, kDLGPU, CuDevice::GetCurrentDeviceId(), 2);
  auto* tensor = &managed_tensor->dl_tensor;

  // DLPack assumes row major, so we use strides[0]
  return new DLPackCuSubMatrix<float>(reinterpret_cast<float*>(tensor->data),
                                      tensor->shape[0], tensor->shape[1],
                                      tensor->strides[0], managed_tensor);
#else
  KALDI_ERR << "Kaldi is not compiled with GPU!";
  return py::none();
#endif
}

}  // namespace kaldi

using namespace kaldi;

void pybind_dlpack(py::module& m) {
  m.def("SubVectorFromDLPack",
        [](py::capsule* capsule) { return SubVectorFromDLPack(capsule); },
        py::return_value_policy::take_ownership);
  // we use `take_ownership` because it returns a pointer created with `new`
  // and we want to transfer the ownership of the pointer to Python.

  m.def("SubMatrixFromDLPack",
        [](py::capsule* capsule) { return SubMatrixFromDLPack(capsule); },
        py::return_value_policy::take_ownership);

  m.def("CuSubVectorFromDLPack",
        [](py::capsule* capsule) { return CuSubVectorFromDLPack(capsule); },
        py::return_value_policy::take_ownership);

  m.def("CuSubMatrixFromDLPack",
        [](py::capsule* capsule) { return CuSubMatrixFromDLPack(capsule); },
        py::return_value_policy::take_ownership);
}

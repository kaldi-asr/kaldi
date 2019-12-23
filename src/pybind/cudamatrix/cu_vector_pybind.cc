// pybind/cudamatrix/cu_vector_pybind.cc

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

#include "cudamatrix/cu_vector_pybind.h"

#include "cudamatrix/cu-vector.h"
#include "dlpack/dlpack_deleter.h"

using namespace kaldi;

void pybind_cu_vector(py::module& m) {
  {
    using PyClass = CuVectorBase<float>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "FloatCuVectorBase", "Vector for CUDA computing")
        .def("Dim", &PyClass::Dim, "Dimensions")
        // the following methods are only for testing
        .def("SetZero", &PyClass::SetZero)
        .def("Set", &PyClass::Set, py::arg("value"))
        .def("Add", &PyClass::Add, py::arg("value"))
        .def("Scale", &PyClass::Scale, py::arg("value"))
        .def("__getitem__", [](const PyClass& v, int i) { return v(i); })
        .def("to_dlpack", [](PyClass* v) {
#if HAVE_CUDA == 1
          // we use the name `to_dlpack` because PyTorch uses the same name

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
          tensor->data = v->Data();
          tensor->ctx.device_type = kDLGPU;
          tensor->ctx.device_id = CuDevice::GetCurrentDeviceId();

          tensor->ndim = 1;

          tensor->dtype.code = kDLFloat;
          tensor->dtype.bits = 32;  // single precision float
          tensor->dtype.lanes = 1;

          // `shape` and `strides` are freed in `DLManagedTensorDeleter`,
          // so no memory leak here.
          tensor->shape = new int64_t[1];
          tensor->shape[0] = v->Dim();

          tensor->strides = new int64_t[1];
          tensor->strides[0] = 1;
          tensor->byte_offset = 0;

          // WARNING(fangjun): the name of the capsule MUST be `dltensor`
          // for PyTorch; refer to
          // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L383/
          // for more details
          return py::capsule(managed_tensor, "dltensor");
#else
          KALDI_ERR << "Kaldi is not compiled with GPU!";
          return py::none();
#endif
        });
  }
  {
    using PyClass = CuVector<float>;
    py::class_<PyClass, CuVectorBase<float>>(m, "FloatCuVector")
        .def(py::init<>())
        .def(py::init<MatrixIndexT, MatrixResizeType>(), py::arg("dim"),
             py::arg("MatrixResizeType") = kSetZero)
        .def(py::init<const VectorBase<float>&>(), py::arg("v"));
    // TODO(fangjun): wrap other methods when needed
  }
  {
    using PyClass = CuSubVector<float>;
    py::class_<PyClass, CuVectorBase<float>>(m, "FloatCuSubVector")
        .def("from_dlpack", [](py::capsule* capsule) {
#if HAVE_CUDA == 1
          DLManagedTensor* managed_tensor = *capsule;

          auto* tensor = &managed_tensor->dl_tensor;

          // we support only 1-D tensor
          KALDI_ASSERT(tensor->ndim == 1);

          // we support only float (single precision, 32-bit) tensor
          KALDI_ASSERT(tensor->dtype.code == kDLFloat);
          KALDI_ASSERT(tensor->dtype.bits == 32);
          KALDI_ASSERT(tensor->dtype.lanes == 1);

          auto* ctx = &tensor->ctx;
          KALDI_ASSERT(ctx->device_type == kDLGPU);
          KALDI_ASSERT(ctx->device_id == CuDevice::GetCurrentDeviceId());

          return CuSubVector<float>(reinterpret_cast<float*>(tensor->data),
                                    tensor->shape[0]);
#else
          KALDI_ERR << "Kaldi is not compiled with GPU!";
          return py::none();
#endif
        });
  }
}

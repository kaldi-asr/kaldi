// pybind/cudamatrix/cu_matrix_pybind.cc

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

#include "cudamatrix/cu_matrix_pybind.h"

#include "cudamatrix/cu-matrix.h"
#include "dlpack/dlpack_deleter.h"

using namespace kaldi;

void pybind_cu_matrix(py::module& m) {
  {
    using PyClass = CuMatrixBase<float>;
    py::class_<PyClass, std::unique_ptr<PyClass, py::nodelete>>(
        m, "FloatCuMatrixBase", "Matrix for CUDA computing")
        .def("NumRows", &PyClass::NumRows, "Return number of rows")
        .def("NumCols", &PyClass::NumCols, "Return number of columns")
        .def("Stride", &PyClass::Stride, "Return stride")
        // the following methods are only for testing
        .def("ApplyExp", &PyClass::ApplyExp)
        .def("SetZero", &PyClass::SetZero)
        .def("Set", &PyClass::Set, py::arg("value"))
        .def("Add", &PyClass::Add, py::arg("value"))
        .def("Scale", &PyClass::Scale, py::arg("value"))
        .def("__getitem__",
             [](const PyClass& m, std::pair<ssize_t, ssize_t> i) {
               return m(i.first, i.second);
             })
        .def("to_dlpack", [](PyClass* m) {
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
          tensor->data = m->Data();
          tensor->ctx.device_type = kDLGPU;
          tensor->ctx.device_id = CuDevice::GetCurrentDeviceId();

          tensor->ndim = 2;

          tensor->dtype.code = kDLFloat;
          tensor->dtype.bits = 32;  // single precision float
          tensor->dtype.lanes = 1;

          // `shape` and `strides` are freed in `DLManagedTensorDeleter`,
          // so no memory leak here
          tensor->shape = new int64_t[2];
          tensor->shape[0] = m->NumRows();
          tensor->shape[1] = m->NumCols();

          tensor->strides = new int64_t[2];
          tensor->strides[0] = m->Stride();
          tensor->strides[1] = 1;
          tensor->byte_offset = 0;

          // WARNING(fangjun): the name of the capsule MUST be `dltensor`
          // for PyTorch; refer to
          // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Module.cpp#L383/
          // for more details.
          return py::capsule(managed_tensor, "dltensor");
#else
          KALDI_ERR << "Kaldi is not compiled with GPU!";
          return py::none();
#endif
        });
  }

  {
    using PyClass = CuMatrix<float>;
    py::class_<PyClass, CuMatrixBase<float>>(m, "FloatCuMatrix")
        .def(py::init<>())
        .def(py::init<MatrixIndexT, MatrixIndexT, MatrixResizeType,
                      MatrixStrideType>(),
             py::arg("rows"), py::arg("cols"),
             py::arg("resize_type") = kSetZero,
             py::arg("MatrixStrideType") = kDefaultStride)
        .def(py::init<const MatrixBase<float>&, MatrixTransposeType>(),
             py::arg("other"), py::arg("trans") = kNoTrans);
    // TODO(fangjun): wrap other methods when needed
  }
  {
    using PyClass = CuSubMatrix<float>;
    py::class_<PyClass, CuMatrixBase<float>>(m, "FloatCuSubMatrix")
        .def("from_dlpack", [](py::capsule* capsule) {
#if HAVE_CUDA == 1
          DLManagedTensor* managed_tensor = *capsule;

          auto* tensor = &managed_tensor->dl_tensor;

          // we support only 2-D tensor
          KALDI_ASSERT(tensor->ndim == 2);

          // we support only float (single precision, 32-bit) tensor
          KALDI_ASSERT(tensor->dtype.code == kDLFloat);
          KALDI_ASSERT(tensor->dtype.bits == 32);
          KALDI_ASSERT(tensor->dtype.lanes == 1);

          auto* ctx = &tensor->ctx;
          KALDI_ASSERT(ctx->device_type == kDLGPU);
          KALDI_ASSERT(ctx->device_id == CuDevice::GetCurrentDeviceId());

          // DLPack assumes row major, so we use strides[0]
          return CuSubMatrix<float>(reinterpret_cast<float*>(tensor->data),
                                    tensor->shape[0], tensor->shape[1],
                                    tensor->strides[0]);
#else
          KALDI_ERR << "Kaldi is not compiled with GPU!";
          return py::none();
#endif
        });
  }
}

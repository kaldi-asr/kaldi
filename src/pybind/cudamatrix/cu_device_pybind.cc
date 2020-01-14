// pybind/cudamatrix/cudamatrix_pybind.cc

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

#include "cudamatrix/cu_device_pybind.h"

#include "base/kaldi-error.h"
#include "cudamatrix/cu-device.h"

using namespace kaldi;

void pybind_cu_device(py::module& m) {
  m.def("SelectGpuDevice",
        [](int device_id) {
#if HAVE_CUDA == 1
          CuDevice::Instantiate().SelectGpuDevice(device_id);
#else
          KALDI_LOG << "Kaldi is NOT compiled with GPU! Ignore it.";
#endif
        },
        py::arg("device_id"));

  m.def("SelectGpuId",
        [](const std::string& use_gpu) {
#if HAVE_CUDA == 1
          CuDevice::Instantiate().SelectGpuId(use_gpu);
#else
          KALDI_LOG << "Kaldi is NOT compiled with GPU! Ignore it.";
#endif
        },
        py::arg("use_gpu"));
  
  m.def("CuDeviceAllowMultithreading", []() {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().AllowMultithreading();
#else
    KALDI_LOG << "Kaldi is NOT compiled with GPU! Ignore it.";
#endif
  });

  m.def("CudaCompiled",
        []() -> bool {
#if HAVE_CUDA == 1
          return true;
#else
          return false;
#endif
        },
        "true if kaldi is compiled with GPU support; false otherwise");
}

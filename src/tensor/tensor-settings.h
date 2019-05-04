// tensor/tensor-settings.h

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
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

#ifndef KALDI_TENSOR_TENSOR_SETTINGS_H_
#define KALDI_TENSOR_TENSOR_SETTINGS_H_ 1

#include <cstdint>
#include <vector>
#include <string>
#include "tensor/tensor-common.h"


/**
   This file contains certain mechanisms to set settings about default
   data types and devices within scopes, some related things like
   an equivalent of PyTorch's .no_grad().  Also the `Tick()` mechanism
   is here.
*/

namespace kaldi {
namespace tensor {


Device GetDefaultDevice();
void SetDefaultDevice(Device device);

// Mechanism to set the default device within a scope by constructing a variable
// that exists only within that scope.
class WithDeviceAs {
 public:
  // Example:
  // {
  //   WithDeviceAs _(kCudaDevice);
  //   // code in this block uses this default.  the variable
  //   // name is _ because we don't need to access it.
  // }
  inline WithDeviceAs(DeviceType device_type):
      prev_default_(GetDefaultDevice()) {
    SetDefaultDevice(Device(device_type));
  }
  inline WithDeviceAs(Device device):
      prev_default_(GetDefaultDevice()) {
    SetDefaultDevice(device);
  }
  ~WithDeviceAs() { SetDefaultDevice(prev_default_); }

 private:
  Device prev_default_;
};



DataType GetDefaultDtype();
void SetDefaultDtype(DataType dtype);

class WithDtypeAs {
 public:
  // Example:
  // {
  //   WithDtypeAs _(kDoubleDtype);
  //   // code in this block uses this default.  the variable
  //   // name is _ because we don't need to access it.
  // }
  inline WithDtypeAs(DataType dtype):
      prev_default_(GetDefaultDtype()) {
    SetDefaultDtype(dtype);
  }
  ~WithDtypeAs() { SetDefaultDtype(prev_default_); }

 private:
  DataType prev_default_;
};



// struct TensorOptions is used as an arg for some constructors
// when creating Tensors and Variables; it allows flexibility
// in specifying the device and/or dtype.  See the examples
// shown where constructors of Tensor or Variable are declared.
struct TensorOptions {
  DataType dtype;
  Device device;

  TensorOptions(): dtype(GetDefaultDtype()),
                   device(GetDefaultDevice()) { }
  TensorOptions(DataType dtype):
      dtype(dtype), device(GetDefaultDevice()) { }
  TensorOptions(Device device):
      dtype(GetDefaultDtype()), device(device) { }
  TensorOptions(DeviceType device_type):
      dtype(GetDefaultDtype()), device(device_type) { }
  TensorOptions(DataType dtype, Device device):
      dtype(dtype), device(device) { }
  TensorOptions(DataType dtype, Device device_type):
      dtype(dtype), device(device_type) { }
  TensorOptions(const TensorOptions &other):
      dtype(other.dtype), device(other.device) { }
};


// Global variable, initialized from zero, that is used in GetTick().
// This is defined in tensor-settings.cc.
extern int64 g_tick_counter;
inline int64 NextTick() { return ++g_tick_counter; }


// debug_mode activates code that checks for invalidated data in the backprop
// pass; see "Invalidated:" in glossary in tensor.h.
// Don't access this variable directly,
extern thread_local bool debug_mode;
inline bool DebugMode() { return debug_mode; }
inline void SetDebugMode(bool b) { debug_mode = b; }

class WithDebugModeAs {
 public:
  // Example:
  // {
  //   WithDebugModeAs _(true);
  //   // code in this block uses debug mode.
  //   // variable name is _ because we won't use it.
  // }
  inline WithDebugModeAs(bool b):
      prev_default_(DebugMode()) {
    SetDebugMode(b);
  }
  ~WithDebugModeAs() { SetDebugMode(prev_default_); }

 private:
  bool prev_default_;
};



// allow_grad means that gradient tracking is allowed; allow_grad = true
// is the normal case, and means that if gradient tracking is required
// (e.g. if the user created a Variable with requires_grad = true, and we do
// operations that depend on it), then we'll track gradients.
// It is our way to implement an equivalent of PyTorch's `with torch.no_grad()`.
// Do not access this variable directly; use AllowGrad() and
extern thread_local bool allow_grad;
inline bool AllowGrad() { return allow_grad; }
inline void SetAllowGrad(bool b) { allow_grad = b; }


class WithNoGrad {
 public:
  // Example:
  // {
  //   WithNoGrad _;
  //   // code in this block has gradient tracking disabled.
  //   // variable name is _ because we won't use it.
  //
  // }
  inline WithNoGrad():
      prev_default_(AllowGrad()) {
    SetAllowGrad(false);
  }
  ~WithNoGrad() { SetAllowGrad(prev_default_); }
 private:
  bool prev_default_;
};


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_SETTINGS_H_

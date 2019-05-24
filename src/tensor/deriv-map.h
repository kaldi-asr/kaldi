// tensor/deriv-map.h

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




namespace kaldi {
namespace tensor {


/*
  Derivative shape:

  For a quantity of shape, say, [ 2 3 ], the derivative will have the exact
  same shape [ 2 3 ] if ExtraDim() == 0, but if ExtraDim == x with x > 0,
  the derivative will have the shape [ x 2 3 ].  This makes it possible
  to compute derivatives w.r.t. vector-value quantities (of course, this
  would be more expensive).

*/
class DerivMap {
 public:
  DerivMap(const DerivMap &other);

  // Default constructor, constructs an empty DerivMap taking the derivative
  // w.r.t a scalar (or of a scalar w.r.t. the things in the forward pass).
  DerivMap();


  // Constructor where you can provide a vector of extra dimensions that the
  // derivatives will have (ordered as in the public numbering, in which
  // they will appear before the dimensions of the things used in the
  // forwardpass).  This is for when you are taking the derivative w.r.t.
  // a more-than-scalar-valued quantity (in backward mode) or taking the
  // derivative of a more-than-scalar-valued quantity w.r.t. things
  // (in forward mode).
  // This should rarely be used.
  DerivMap(conststd::vector<int32> &extra_dims);

  // Returns the derivative Tensor for Tensor 't', if one exists already; else
  // NULL.  (To explain return type, see "Optional Tensor" in tensor.h).
  std::shared_ptr<TensorImpl> DerivIfPresent(const Tensor &t) const;

  /**
     Returns the derivative for Tensor t, creating it if it did not already
     exist.  The mapping from t to its derivative is only stored in this class.
     See "Derivative shape:" above for explanation of the shape of this Tensor;
     it will usually be the same as the shape of t.
     In order to make sure that a Tensor t has an entry in this DerivMap,
     you can call this function and ignore the return value.

     Note: the derivative objects are created at the level of the Storage
     region, so when any Tensor that uses a particular storage region
     becomes tracked, all other Tensors using that storage region also
     become tracked.

         @param [in] t  The Tensor whose derivative the user is requesting
  */
  Tensor Deriv(const Tensor &t);


  /**
     Must be called when the DerivMap is empty; set the dimension of the
     quantity that we are computing the derivative of (or with respect to).
     Would be 0 in most situations, meaning the derivative is w.r.t.  a scalar,
     but if it is >0, the derivatives returned by this DerivMap will have an
     extra dmension (search for "Derivative shape" above).
  */
  void SetExtraDim(int32 extra_dim);


  /**
     Returns a value that is always positive and normally 1, which is the product of extra_dims_.
  */
  int64 ExtraDimsProd() const  { return extra_dims_prod_; }

  std::vector<int32> &ExtraDims() const  { return extra_dims_; }

 private:

  // extra_dims_ is the shape (in the public numbering) of the thing that we are taking
  // the derivative of (in backward mod) or with respect to (in forward mode).
  // It would normally be the empty vector, meaning we're taking the derivative
  // w.r.t. a scalar.  All elements must be positive.
  std::vector<int32> extra_dims_;
  // extra_dims_prod_ is the product of the elements of extra_dims_.
  // It will normally be 1.
  int64 extra_dims_prod_;


  // The record relating to the map from one source Storage object to the
  // corresponding derivative.  The num_bytes of the deriv_storage object will
  // be equal to the num_bytes of src_storage times extra_dims_prod_.
  struct DerivRecord {
    std::weak_ptr<Storage> src_storage;
    std::weak_ptr<Storage> deriv_storage;
  };

  // The key in this map is the int64 tick value when the src Storage
  // object was created (see its Id() function).
  // The value
  std::unordered_map<int64, DerivRecord> map_;


};


// class Context contains various configurations that we will sometimes need
// when we do operations on Tensors.  Things like the default data type, the
// debug mode, and so on.  This will be passed around
class Context {

};

class AutogradContext: public Context {
 public:


 private:
  DataType default_dtype_;
  Device default_device_;


  std::shared_ptr<


  bool store_ops_;

};


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
extern bool debug_mode;     // Do not access directly!
extern int64 debug_start_tick;   // Do not access directly!

inline bool DebugMode() {
  return debug_mode;
}
inline void SetDebugMode(bool b) {
  if (!debug_mode)
    debug_start_tick = NextTick();
  debug_mode = b;
}
/**
   Returns the tick at which debug mode most recently changed from false to
   true.
 */
inline int64 DebugTick() {
  KALDI_PARANOID_ASSERT(debug_mode);
  return debug_start_tick;
}

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

// tensor/context.h

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

#ifndef KALDI_TENSOR_CONTEXT_H_
#define KALDI_TENSOR_CONTEXT_H_ 1

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


// class Context contains various configurations that we will sometimes need
// when we do operations on Tensors.  Things like the default data type, the
// debug mode, and so on.  This will be passed around
class Context {
  // The default DataType for newly created Tensors
  DataType default_dtype_;
  // The default Device for newly created Tensors
  Device default_device_;
};



// ExecutionContext is used when executing Ops (or doing other things
// with them, e.g. just storing them); we explicitly pass this
// object into functions that might want to execute Ops.
class ExecutionContext: public Context {

  /// This function executes the Op (op.Do()) and/or does something else
  /// relating to taking derivatives.
  virtual void Execute(const Op &op);

  virtual ~ExecutionContext() {}
};


// SimpleExecutionContext means we just execute an Op and then immediately
// delete it.  It's used when we are just computing something with no
// autograd.  You could, of course, just call the version of the
// Op that doesn't take an ExecutionContext, but this option makes
// it easier to switch between autograd and no-autograd.
class SimpleExecutionContext: public ExecutionContext {

  virtual void Execute(const Op &op) {  op.Do();  }
  virtual ~SimpleExecutionContext() {}
};




/**
   Execution context that you use while doing a forward computation, that
   executes the forward commands and stores the things required to later do the
   backprop.  See its Backprop() function for how to execute the backprop.
*/
class BackpropExecutionContext: public ExecutionContext {

  /**
     Constructor of BackpropExecutionContext from an existing DerivMap, which
     might map, for instance, parameters to their derivatives.

      @param [in] deriv_map   An existing DerivMap, to which the user will
                      likely have added the model parameters and anything
                      else that derivatives are needed for, with its
                      Deriv() function.  This is *copied*, not held as a
                      reference, by this object, to avoid a kind of memory
                      leakage.
      @param [in] base_context  The base execution context, which would
                      normally be SimpleExecutionContext; it is used to
                      execute both the forward and backward commands.
                      This class will store the pointer but will not take
                      ownership; it is the user's responsibility to
                      make sure it stays alive as long as this object is
                      alive.
   */
  BackpropExecutionContext(const DerivMap &deriv_map,
                           ExecutionContext *base_context);




  /**
     Does the backprop on a Tensor t; propagates the derivative back to whatever
     quantities you had added derivs for in the DerivMap passed to the constructor.

     The backprop commands will be executed with a SimpleExecutionContext
     whose Context base-class is a copy of this class's one.  If you want to
     do something fancier (e.g. for 2, you can use the version of Backprop

     If retain_info is false, it will delete deriv_map_ and clear backward_ops_.
     This is recommended in most cases; it's more memory efficient.

        @param [in] t    The Tensor that we are taking the derivative with
                        respect to.
        @param [in] deriv  The derivative w.r.t. t of the function we
                        are taking the derivative of.  Might be just
                        1.0.  Must satsify Broadcastable(deriv, t).
                        Note: deriv may have more axes than t, in which
                        case the extra leading axes are required to
                        have dimensions equal to deriv_map_->ExtraDims().
                        If deriv_map_->ExtraDims() is nonempty,
                        the num-axes of 'deriv' is required to equal
                        `t.NumAxes() + deriv_map_->ExtraDims().size()`.
   */
  void Backprop(const Tensor &t,
                const Tensor &deriv) {
    if (deriv_map_ == nullptr)
      KALDI_ERR << "You cannot call Backprop twice on the same "
          "BackpropExecutionContext";

    // Delete deriv_map_.  This will help ensure that derivative
    // quantities are deleted as soon as they are no longer needed
    // (since once we delete the deriv_map_ and the ops referring
    // to those derivative matrices, they will be garbage collected).
    deriv_map_ = nullptr;

    for (auto iter = backward_ops_.rbegin();
         iter != backward_ops_.rend(); ++iter){
      base_context_->Execute(**iter);
      // Delete this op.  Deleting the ops also deletes the associated
      // derivative matrices, via shared_ptr garbage collection.
      *iter = nullptr;
    }
    backward_ops_.clear();
  }


  virtual void Execute(const Op &op) {
    base_context_->Execute(op);
    op.GetBackwardDerivOps(&deriv_map_, &backward_ops_);
  }

  virtual ~BackpropExecutionContext() { }


 private:
  std::vector<unique_ptr<Op> > backward_ops_;
  unique_ptr<DerivMap> deriv_map_;
  ExecutionContext *base_context_;

};


/**
   Execution context that you use while doing a forward computation, that
   executes the forward commands and also computes forward derivatives
   w.r.t. something.
*/
class ForwardPropExecutionContext: public ExecutionContext {

  /**
     Constructor of ForwardPropExecutionContext from an existing DerivMap, which
     might map, for instance, some input x to dx/da, where a is the thing
     we're taking the derivative of.

      @param [in] deriv_map   An existing DerivMap, to which the user will
                      likely have added the thing we are taking the derivative
                      w.r.t. (e.g. some input where we want to see its
                      effect on the computation).  deriv_map is *copied*,
                      not held as a reference, by this object, to avoid
                      a kind of memory leakage.
      @param [in] base_context  The base execution context, which would
                      normally be SimpleExecutionContext; it is used to
                      execute both the forward and backward commands.
                      This class will store the pointer but will not take
                      ownership; it is the user's responsibility to
                      make sure it stays alive as long as this object is
                      alive.
   */
  ForwardPropExecutionContext(const DerivMap &deriv_map,
                              ExecutionContext *base_context);


  virtual void Execute(const Op &op) {
    base_context_->Execute(op);
    std::vector<std::unique_ptr<Op> > ops;
    op.GetForwardDerivOps(&deriv_map_, &ops);
    for (auto iter = ops.begin(); iter != ops.end(); ++iter)
      base_context_->Execute(*iter);
    // and let the ops in 'ops' go out of scope and get deleted.
  }


  // Returns pointer to this deriv_map_ (still owned by this class).
  // May be used to query the derivative of some Tensor w.r.t. the
  // input, e.g. forward_context.GetDerivMap()->DerivIfPresent(some_tensor).
  DerivMap *GetDerivMap() { return deriv_map_.get(); }


};





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


#endif  // KALDI_TENSOR_CONTEXT_H_

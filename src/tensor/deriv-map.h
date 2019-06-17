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
  class DerivMap stores and updates a map from a Tensor to some derivative
  quantity related to that Tensor.  We store this map separately from
  the Tensor itself because this seems to generalize more naturally
  to things like higher-order derivative, and helps keep the code
  easy to understand.

  Note: the memory for the derivatives is actually allocated for the whole
  Storage region underlying a Tensor, so if we call Deriv() to create the
  derivative for some Tensor, all Tensors sharing the same underlying storage
  region will now also have an entry in the DerivMap.

  Derivative shape:

  For a quantity of shape, say, [ 2 3 ], the derivative will normally have the exact
  same shape, e.g. [ 2 3 ].  But if the extra_dims


  if ExtraDim() == 0, but if ExtraDim == x with x > 0,
  the derivative will have the shape [ x 1 1 1 2 3 ].  This makes it possible
  to compute derivatives w.r.t. vector-value quantities (of course, this
  would be more expensive).
*/
class DerivMap {
 public:
  /** Construct a new, empty DerivMap.
       @param [in] context  Context that determinize the dtype and device
                     for derivatives we create.
  */
  DerivMap(const Context &context);


  /**
     Constructor where you can provide a vector of extra dimensions that the
     derivatives will have (ordered as in the public numbering, in which
     they will appear before the dimensions of the things used in the
     forwardpass).  This is for when you are taking the derivative w.r.t.
     a more-than-scalar-valued quantity (in backward mode) or taking the
     derivative of a more-than-scalar-valued quantity w.r.t. things
     (in forward mode).  This should rarely be used.

        @param [in] context  Object that sets the default device and dtype
        @param [in] extra_dims   Extra dimensions, ordered as in the public
                       numbering, that the derivative has, e.g. in reverse-mode
                       autograd (backprop) this is used when we are taking
                       derivatives w.r.t a non-scalar quantity.
        @param [in] axis_offset  The user should set this to a number >= the
                       largest num_axes of any of the Tensors with which
                       we will call Deriv() or DerivIfPresent() with this
                       object.  (Note: any matrix multiplication implicitly
                       adds an axis, so for example if you are doing matrix
                       multiplication on Tensors with 3 axes, you should
                       make sure axis_offset is at least 4) axis_offset
                       ensures that the 'extra_dims' always appear
                       at the same position regardless of the num_axes
                       of the Tensor we called Deriv() with.
                       Technically, axis_offset is only an axis offset in
                       the private numbering;; in the public numbering it's the
                       num_axes to which we pad the Tensors supplied to Deriv()
                       before prepending extra_dims.

     Example: if extra_dims = [2 3] and axis_offset = 4, and someone calls
     Deriv() with a Tensor of shape [7 8], the derivative Tensor will have
     shape [2 3 1 1 7 8].  (Note: any unused/trivial axes will have no effect
     on the actual computation).
  */
  DerivMap(const Context &context,
           ArrayRef<int32> extra_dims,
           int32 axis_offset);


  /**
     Copy constructor.  This is expected to be used in typical neural net
     training workflows, where we create a DerivMap for the parameters, and then
     use it with the copy constructor to initialize a fresh DerivMap that will
     also store the derivatives for the temporary quantities.
  */
  DerivMap(const DerivMap &other);



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
     Returns a value that is always positive and normally 1, which is the product of extra_dims_.
  */
  int64 ExtraDimsProd() const  { return extra_dims_prod_; }

  std::vector<int32> &ExtraDims() const  { return extra_dims_; }

 private:

  Context context_;  // Dictates default dtype and device.

  // extra_dims_ is the shape (in the public numbering) of the thing that we are
  // taking the derivative of (in backward mod) or with respect to (in forward
  // mode).  It would normally be the empty vector, meaning we're taking the
  // derivative w.r.t. a scalar.  All elements must be positive.
  std::vector<int32> extra_dims_;

  // determines where we place the extra_dims_ (in the private numbering); or,
  // in the public numbering, what num-axes we pad the arg to Deriv to, before
  // prepending the dims in extra_dims_.   See example given in the doc
  // for the 3-arg constructor.
  int32 axis_offset_;

  // extra_dims_prod_ is the product of the elements of extra_dims_.  It will
  // normally be 1.
  int64 extra_dims_prod_;


  // The record relating to the map from one source Storage object to the
  // corresponding derivative.  The num_bytes of the deriv_storage object will
  // be equal to the num_bytes of src_storage times extra_dims_prod_.
  struct DerivRecord {
    std::weak_ptr<Storage> src_storage;
    std::weak_ptr<Storage> deriv_storage;
  };

  // The key in this map is the int64 tick value when the src_storage object was
  // created (see its Id() function).  (We don't use its memory address, since
  // those can be re-used).
  std::unordered_map<int64, DerivRecord> map_;
};




}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_SETTINGS_H_

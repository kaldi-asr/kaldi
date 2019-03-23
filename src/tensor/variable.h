// tensor/variable.h

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

#ifndef KALDI_TENSOR_TENSOR_H_
#define KALDI_TENSOR_TENSOR_H_ 1

#include "tensor/variable.h"

namespace kaldi {
namespace tensor {


/*
  This is the 'gradient information' that class Variable stores for a Tensor
  when it is initialized with requires_grad = true (or is a result of
  an operation on Variables one of which had requires_grad = true).
  This does not give you access to the underlying Variables; doing it
  like this makes reference counting easier (no loops).  The GradFunc
  will store any pointers to the original Variable that it may have
  needed.

  Users will rarely need to interact directly with this struct directly.
 */
struct TensorGrad {
  // The gradients corresponding to the input variables, which
  // we may need to update.  Some subset of these may be nullptr,
  // corresponding to input Variables for which no gradient
  // was required.
  std::vector<std::shared_ptr<TensorGrad> > inputs;

  // is_view is
  bool is_view{false};

  // The device we
  Device device;

  // The dimension of the Tensor for which this is the gradient.  Used
  // to set up 'grad' when needed.
  TensorPattern dim;

  // 'offset' is only inspected if this is a view; it is the offset
  // (in elements) from the
  // 'inputs' will just contain one member, which is the gradient for the source
  // Variable, and we use 'dim' and 'offset' to construct the sub-tensor).
  int32 offset;

  // This stores the gradient (if we already have one), or nullptr if not.
  std::unique_ptr<Variable> grad{nullptr};

};


/**
   class Variable is somewhat like class Tensor but augmented with autograd
   machinery.  Because autograd requires a rather 'functional' way of doing
   things (i.e. is not super friendly to in-place operations), the functions
   that operate on class Variable will tend to be ones that return something,
   rather than in-place operations.

   The overall design is quite similar to PyTorch, and the structure
   of the the C++ code is similar to flashlight.  If you are only familiar with
   PyTorch's python frontend, class Variable is rougtly equivalent to what they
   expose as af.tensor.
 */
class Variable {
  using GradFunc = std::function<
    void(const std::vector<Variable>& inputs, TensorGrad *grad_output)>;
  using GradHook = std::function<void(TensorGrad *grad)>;



  /** Constructor from a Tensor.
       @param [in] data  Pointer to the source Tensor
       @param [in] requires_grad    If requires_grad argument is true,
                the gradient w.r.t. this Variable will be computed if and when
                you call Backward() on a Variable that depends on it.
                The same as requires_grad in PyTorch.
  */
  Variable(const std::shared_ptr<Tensor> &data, bool requires_grad);



  /**
   * Creates a Variable which wraps the array and inputs specified
   * @param[in] data array to the stored in the Variable
   * @param[in] inputs a vector specifying inputs for this Variable
   * @param[in] gradFunc function specifying how to calculate gradient of the
   * input Variables
   */
  Variable(std::shared_ptr<Tensor> &data, std::vector<Variable> inputs,
           GradFunc gradFunc);



};

typedef std::unique_ptr<Storage>




};


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_VARIABLE_H_

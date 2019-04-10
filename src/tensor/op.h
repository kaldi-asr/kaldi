// tensor/op.h

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

#ifndef KALDI_TENSOR_TENSOR_OP_H_
#define KALDI_TENSOR_TENSOR_OP_H_ 1

#include "tensor/tensor.h"

namespace kaldi {
namespace tensor {

class Variable;


/**
   class Op is a base-class for objects that are created when we compute
   functions of Variables; they exist as long as we retain the computation
   graph.  In fact, the Ops (together with the Variables) *are* the
   computation graph.  An op may in general have multiple input Variables
   and multiple output Variables.

   Each Variable that has gradient tracking and that is not an input Variable
   (i.e. a leaf node) keeps an Op that created it.  Variables that had in-place
   operations done may have more than one Op; these form a singly linked list
   with an ordering (c.f. Op::Tail()).

   When a user calls Backprop() on a Variable, the backprop code works out a
   topological order of Ops and calls the Ops in (essentially) the reverse order
   in which they were created.  The backprop code also frees gradients of
   Variables when it knows they will no longer be needed.


 */
class Op {
 public:

  Op(): n_(GetCount()) { }

  /// InputIteratorBegin() and InputIteratorEnd() form the begin and
  /// end points of a list of Variables that were inputs of this Op
  /// but not outputs.  This is used by the backprop code when finding
  /// the topological order of ops.  (Note: output variables themselves
  /// refer to Ops, so if we included them in the input list we'd
  /// get a cycle in the graph).  These Variables are expected to
  /// still have their graph information (i.e. sub-classes of this
  /// class must not call RemoveGraph() on the members of this list).
  virtual Variable *InputIteratorBegin() = 0;
  virtual Variable *InputIteratorEnd() = 0;



  Op *GetTail() final;  // returns the tail (in a singly linked list of Ops for this
                  // variable); this list will only have >1 element if in-place
                  // operations were done.  (If, later, we need the shared_ptr
                  // to be returned from here, we can change this code to return
                  // that; we just return the raw pointer for efficiency.)


  // Connect this Op to Variable 'v', which is expected to be an output of this
  // Op.  What this does is to ensure that 'v' has a TensorGrad object,
  // and add this Op as the head of the 'ops_' list of v.
  void ConnectToOutput(Variable *v) final;

  // Checks that tail_ is currently nullptr, and sets tail_ to 'op'.
  void SetTail(std::shared_ptr<Op> op) final;

  // This number is used to determine the order of Ops in a graph; each time we
  // generate an Op we increment a global counter.  Doing it this way, rather
  // than via topological sorting, avoids the need to think about having to
  // track implicit dependencies such as when different Ops operate on Variables
  // that are overlapping views into an underlying Variable.
  int64 GetNumber() const final { return n_; }

  virtual void Backward();

 private:
  static int64 counter_{0};
  static int64 GetCount() { return counter_++; }
  int64 n_;
  std::shared_ptr<Op> tail_{nullptr};


};


class AddToOp: public Op {
 public:

  // This Op corresponds to the computation:
  //   \f$  b  :=  alpha a  +   beta b.  \f$
  // with broadcasting or summation depending on the dimensions
  // involved.  Obviously alpha and beta are constants,
  // and differentiation w.r.t. them is not supported.
  //
  // The Op is only constructed if a_.HasGrad() || b_.HasGrad().
  AddToOp(float alpha, float beta,
          const Variable &a, const Variable *b):
      alpha_(alpha),
      beta_(beta),
      a_(a, kCopyVarGrad|kCopyVarGraph),
      b_(b, kCopyVarGrad),
      must_scale_b_grad_(beta != 1.0 && b->HasGrad()) {
    AddTo(alpha, beta, a, b);
    this->ConnectToOutput(b);
  }


  void Backward() {
    const Tensor &a_grad = a_->GetGrad(),
        &b_grad = b_->GetGrad();
    // Do: a_grad += alpha * b_grad
    AddTo(alpha_, 0.0, b_grad, &a_grad);
    if (must_scale_b_grad_)
      Scale(beta_, &b_grad);
  }


  // There is only one input that is not also an output: a_.
  Variable *InputIteratorBegin() override { return &a_; }
  Variable *InputIteratorEnd() override { return (&a_) + 1; }

 private:

  float alpha_;
  float beta_;

  Variable a_;
  Variable b_;
  bool must_scale_b_grad_;

};



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_VARIABLE_H_

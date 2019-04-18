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

   Every base Variable (see variable.h for definition) that is tracked
   has a singly linked list of Ops that changed that base Variable,
   ordered from most recent to least recent.

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
  /// but were not outputs.  This is used by the backprop code when finding
  /// the topological order of ops.  (Note: output variables themselves
  /// refer to Ops, so if we included them in the input list we'd
  /// get a cycle in the graph).  These Variables are expected to
  /// still have their graph information (i.e. sub-classes of class Op
  /// class must not call RemoveGraph() on the members of this list).
  virtual Variable *InputIteratorBegin() = 0;
  virtual Variable *InputIteratorEnd() = 0;



  Op *GetTail() final;  // returns the tail (in a singly linked list of Ops for
                        // this variable); this list will only have >1 element
                        // only if in-place operations were done.  (If, later,
                        // we need the shared_ptr to be returned from here, we
                        // can change this code to return that; we just return
                        // the raw pointer for efficiency.)


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

 protected:
  // Constructor, to be used from child classes.  This base-class takes care
  // of storing the list of input Variables for purposes of tracing dependencies;
  //
  //  @param [in] input_vars  The list of input Variables (meaning: Variables
  //                   that are inputs to, but not outputs of, i.e. not modified
  //                   by, this Op).
  //  @param [in] output_var  The output Variable of this Op, i.e. the Variable
  //                   which is modified or set by it.  We will provide another
  //                   constructor taking ArrayRef<Variable> in this position,
  //                   as and when we need to support Ops that take multiple
  //                   output Variables.
  void Op(const ArrayRef<Variable> &input_vars,
          const Variable &output_var);


  // TODO: maybe have a constructor of Op that takes an ArrayRef of the inputs
  // that are not also outputs?  Could use that for graph traversal.

 private:
  static int64 counter_{0};

  // num_inputs_ is the number of base Variables that are the base Variables of
  // inputs of this Op (but not of outputs).  These are stored in the
  // array 'inputs_'.

  // inputs_ is a pointer to an array of shared_ptr<Variable> of size num_inputs_, which
  // will be be allocated by new [] in the constructor and deleted by delete []
  // in the destructor.
  // This is a list of the unique Nodes that are the Nodes of inputs (but not outputs)
  // of this
  std::shared_ptr<Node> *inputs_;

  int32 num_inputs_;
  void *inputs_;

  int64 n_;  // initialized from the counter when this object is created.
  std::shared_ptr<Op> tail_;
 protected:
  // Return true if this is not the last Op in the list of Ops attached to this
  // base Variable (can be useful to know whether we need bother to scale the
  // derivative in a scaling operation, for instance).
  bool HasTail() const { return tail_ != nullptr; }
};


class AddToOp: public Op {
 public:

  // This Op corresponds to the computation:
  //   \f$  b  :=  alpha a  +   beta b.  \f$
  // with broadcasting or summation depending on the dimensions
  // involved.  Obviously alpha and beta are constants,
  // and differentiation w.r.t. them is not supported.
  //
  // The Op is only constructed if b_.HasGrad() (which it
  // would normally if a_.HasGrad()).
  AddToOp(float alpha, float beta,
          const Variable &a, const Variable &b):
      Op({a}),
      alpha_(alpha),
      beta_(beta),
      a_data_(a.GetData()),
      a_grad_(a.GetGradIfPresent()),
      b_data_(b.GetData()),
      b_grad_(b.GetGrad()) {

    Add(alpha, beta, *a_data_, b_data_.get());
  }


  void Backward() {
    // Do: a_grad += alpha * b_grad.
    if (a_grad_ != nullptr)
      AddTo(alpha_, 1.0, b_grad, &a_grad);

    if (beta_ != 1.0)
      Scale(beta_, b_grad.get());
  }

 private:

  float alpha_;
  float beta_;

  // We hold onto all inputs that are not also outputs
  // (here just a_) for dependency tracking.
  Variable a_;

  std::shared_ptr<Tensor> a_data_;
  std::shared_ptr<Tensor> a_grad_;
  std::shared_ptr<Tensor> b_data_;
  std::shared_ptr<Tensor> b_grad_;

  Variable b_;
  bool must_scale_b_grad_;

};



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_VARIABLE_H_

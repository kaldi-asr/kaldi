// tensor/tensor-linear-ops.h

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

#ifndef KALDI_TENSOR_TENSOR_LINEAR_OPS_H_
#define KALDI_TENSOR_TENSOR_LINEAR_OPS_H_ 1

#include "tensor/tensor.h"


// Note: user-level code will not interact directly with these Ops.  See
// tensor-linear.h for the user-level code.
namespace kaldi {
namespace tensor {


/**
   Add operation taking two Tensors (T), i.e. b += a, which
   may include summation and/or broadcasting depending on
   the dimensions of b and a.

   May not be used if b and a overlap.  Probably should not be used if b is
   known to be zero (e.g. if it hasn't been allocated)-- in that case, it's
   better to use CopyOp.
*/
class AddTTOp {
 public:

  AddTTOp(const Tensor &a, Tensor &b):
      a_(a), b_(b) {
    KALDI_ASSERT(!Overlap(a, b) &&
                 BroadcastableAndCompatible(a, b));
  }
  void Do() const override {

  }


 private:
   Tensor a_;
   Tensor b_;
};


/**
   class Op is a base-class for objects that are created when we do operations
   on Variables.  The important thing to know here is that the Variables in
   question will always have been allocated with particular dimensions,
   and possibly even contain defined values, before we get to the Op.
   Examples of Ops include,
      a := b * c
      a += b
      a *= b
   where the interpretation of the commands above will depend on the
   dimensions of the Tensors involved.

   Notice that all the member functions of class Op are `const`, i.e. they
   shouldn't change this class (although of course they may change the
   underlying Tensor data).  This is to remind users that Ops are supposed
   to be reusable, and calls to this object shouldn't affect the behavior
   of subsequent calls, except to the extent that the underlying Tensor
   data has been changed.
 */
class Op {
 public:

  /**
     Do whatever it is that this Op does (e.g. execute the command `a += b`,
     if that was what this Op did)
   */
  virtual void Do() const;

  /**
     Return a copy of this object.  (This won't be needed very often but might
     possibly be needed in the context of computing higher-order derivatives).
  */
  virtual Op *Copy() const;

  /**
     This is for forward-mode automatic differentiation (a rarely-used thing).
     It appends to 'ops' the commands corresponding to the forward-mode
     automatic differentiation w.r.t. this Op.

       @param [in,out] 'map' is the map that maps from tensors to the
             corresponding derivative values.  May be modified by adding
             new key/value pairs.
       @param [out] ops  This funtion will *append* to `ops` the
             commands for computing the derivatives associated with
             this Op in forward-mode automatic differentiation.  If none
             of the inputs to the Op were tracked w.r.t. `map`,
             nothing will be done.

     Example: if the command was "a += b", the derivative operation would
     be: deriv(a) += deriv(b).  In most cases these Ops would be executed
     immediately and then deleted.
   */
  virtual void GetForwardDerivOps(DerivMap *map,
                                  std::vector<std::unique_ptr<Op> > *ops) const;



  /**
     This is for reverse-mode automatic differentiation (the normal type of
     autograd).

       @param [in,out] map   This object maps from tensors to the
                       corresponding derivative values.  It may be changed by
                       adding new elements to the map, if its Deriv() function
                       is called.
       @param [out]    ops  This function may *append* to 'ops' the commands
                       used in the reverse-mode automatic differentiation.
                       (Note: nothing will be appended if none of the inputs
                       to the Op were already tracked w.r.t. 'map'.)

     Example: if the command was "a += b * c", the operations added to
     'ops' would correspond to `deriv(b) += deriv(a) * c` and
     `deriv(c) += deriv(a) * b`.
  */
  virtual void GetBackwardDerivOps(DerivMap *map,
                                   std::vector<std::unique_ptr<Op> > *ops) const;


  /** Destructor.  It's important for efficiency of memory use to destroy Ops as
      soon as you won't need them any more, because it may trigger the freeing
      of Tensors and hence Storage regions.
  */
  virtual ~Op();
};



class Op {

  Op(): tick_(GetTick()) { }

  /// InputIteratorBegin() and InputIteratorEnd() form the begin and
  /// end points of a list of Variables that were inputs of this Op
  /// but were not outputs.  This is used by the backprop code when finding
  /// the topological order of ops.  (Note: output variables themselves
  /// refer to Ops, so if we included them in the input list we'd
  /// get a cycle in the graph).  These Variables are expected to
  /// still have their graph information (i.e. sub-classes of class Op
  /// class must not call RemoveGraph() on the members of this list).
  virtual Op *DepIteratorBegin() = 0;
  virtual Op *DepIteratorEnd() = 0;



  // This number >= 0 is used to determine the order of Ops in a graph; each
  // time we generate an Op we increment a global counter.  Doing it this way,
  // rather than via topological sorting, is simpler.
  int64 GetTimestamp() const final { return tick_; }

  virtual void Backprop();

 protected:

  /**
     The time (`GetTick()`) at which this Op was created; should be set
     in child classes by doing:
      `tick_ = GetTick()`
     as the last statement of the constructor.   (This ensures the
     tick is later-numbered than any ticks stored in the ChangeTracker
     code by operations called from the constructor.)
  */
  int64 tick_;


  /*
    This function intended to be called from the Backprop() routines
    of child classes, for example:
       ` if (DebugMode()) {  CheckTensorTime(*a_);  } `
    This will die if the memory underlying the Tensor being checked has been
    modified more recently than tick_.
  */
  inline void CheckTensorTime(const Tensor &tensor) {
    if (DebugMode()) {
    }
  }




};


template <class OpImpl>
class OpPointer {

  std::shared_ptr<OpImpl>

}



/**
   This is a special version of base-class Op that is created when
   any SharedGrad is allocated for a non-leaf Variable.  Its purpose
   is to ensure that, when we get to this Op in the backprop, we deallocate
   the data underlying the gradient Tensor (so we don't keep gradient
   Tensors around for longer than is needed).
*/
class DeallocateOp: public Op {

  // This operator has no dependencies as it will be created when a SharedGrad
  // is first initialized, when no Ops have been done on it.
  Op *DepIteratorBegin() override { return NULL; }
  Op *DepIteratorEnd() override { return NULL; }

  void Backprop() override {
    if (auto s = tensor_to_deallocate_.lock())
      ZeroDeallocating(s.get());
  }

 private:
  // Since we just want to deallocate its underlying data, there is no point
  // increasing its ref-count; we can just shrug our shoulders if it has
  // already been deleted.d
  std::weak_ptr<Tensor> tensor_to_deallocate_;
};


/**
   A slight simplification of class UnaryOp for cases where it's
   done in-place.
 */
class InPlaceUnaryOp: public Op {

};


class UnaryOp: public Op {

  //
  UnaryOp(const Variable &input, const Variable &output) {
    if



    if (SameVariable(input, output)) {

    } else {
    }
  }

 public:

  std::shared_ptr<Op> op1_;
  std::shared_ptr<Op> op2_;




}

class GenericOp: public Op {

  // GenericOp is a child of class Op that is intended as a generic base-class
  // for expressions.



 protected:
  // Constructor, to be used from child classes.  This base-class takes care
  // of storing the list of input Variables for purposes of tracing dependencies;
  //
  //  @param [in] input_vars  The list of input Variables (meaning: Variables
  //                   that are inputs to, but not outputs of, i.e. not modified
  //                   by, this Op).
  //  @param [in] output_var  The output Variable of this Op, i.e. the Variable
  //                   which is modified or set by it.  We may provide another
  //                   constructor taking ArrayRef<Variable> in this position,
  //                   as and when we need to support Ops that operate on
  //                   multiple output Variables.
  void Op(const ArrayRef<Variable> &input_vars,
          const Variable &output_var);


  // TODO: maybe have a constructor of Op that takes an ArrayRef of the inputs
  // that are not also outputs?  Could use that for graph traversal.

 private:

  // num_inputs_ is the number of base Variables that are the base Variables of
  // inputs of this Op (but not of outputs).  These are stored in the
  // array 'inputs_'.

  // inputs_ is a pointer to an array of shared_ptr<Variable> of size num_inputs_, which
  // will be be allocated by new [] in the constructor and deleted by delete []
  // in the destructor.

  // This is a list of the Op-input-nodes (see glossary in tensor.h for explanation).
  // We don't store the Op-output-nodes here; instead, they refer to this Op in
  // their op_lists.
  // (We don't store the Node(s) that is(are) the outputs of the Op here; its own
  // op_list refers to this Op).
  std::shared_ptr<Node> *inputs_;

  int32 num_inputs_;

  // If num_inputs_ is 1, then inputs_ is
  void *inputs_;

  int64 n_;  // initialized from the counter when this object is created.
  std::shared_ptr<Op> tail_;  // TODO: make it unique_ptr?
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
  // involved.  Alpha and beta are constants, and differentiation w.r.t. them is
  // not supported (you wouldn't reach this code if a or b were actual
  // variables.)
  //
  // The Op is only constructed if b.Tracked() (which it would normally if
  // a.Tracked()).
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

  std::shared_ptr<Node> a_node_;

  std::shared_ptr<Tensor> a_data_;
  // a_grad_ will be NULL if a was not tracked.
  std::shared_ptr<Tensor> a_grad_;
  std::shared_ptr<Tensor> b_data_;
  std::shared_ptr<Tensor> b_grad_;

  Variable b_;
  bool must_scale_b_grad_;

};


class CopyOp: public Op {
 public:

  // This Op corresponds to the computation:
  //   \f$  b := a  \f$
  // with broadcasting or summation depending on the dimensions.
  //
  // Constructing this Op will make b tracked if it was already.
  CopyOp(const Variable &a, const Variable &b):
      Op({a}),
      a_data_(a.GetData()),
      a_grad_(a.GetGradIfPresent()),
      b_data_(b.GetData()),
      b_grad_(b.GetGrad()) {
    Copy(a_data_, b_data_);

      `tick_ = GetTick()`
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

  std::shared_ptr<Node> a_node_;

  std::shared_ptr<Tensor> a_data_;
  // a_grad_ will be NULL if a was not tracked.
  std::shared_ptr<Tensor> a_grad_;
  std::shared_ptr<Tensor> b_data_;
  std::shared_ptr<Tensor> b_grad_;

  Variable b_;
  bool must_scale_b_grad_;

};


class CopyOp: public Op {
 public:

  // This Op corresponds to the computation:
  //   \f$  b  :=  alpha a  +   beta b.  \f$
  // with broadcasting or summation depending on the dimensions
  // involved.  Obviously alpha and beta are constants,
  // and differentiation w.r.t. them is not supported.
  //
  // The Op is only constructed if b_.Tracked() (which it
  // would normally if a_.Tracked()).
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

  std::shared_ptr<Node> a_node_;

  std::shared_ptr<Tensor> a_data_;
  // a_grad_ will be NULL if a was not tracked.
  std::shared_ptr<Tensor> a_grad_;
  std::shared_ptr<Tensor> b_data_;
  std::shared_ptr<Tensor> b_grad_;

  Variable b_;
  bool must_scale_b_grad_;

};



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_LINEAR_OPS_H_

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


/**

   Definitions:

     Tracked:   A variable v is tracked if v.Grad() returns non-NULL.  By "tracked" we mean:
                we are keeping autograd history.  Being tracked or not is actually a
                property of the base variable (see "Base variable").
    Debug mode:  Debug mode is a global bool, accessible via GetDebugMode().  When it is
                true, we check for invalidated data in the backprop phase.  (This,
                which performs the same function as version numbering in, say, PyTorch,
                is quite a slow operation so we only enable it occasionally.
   Base variable: A base Variable is a Variable that is not a sub-part (e.g. row or
                column range) of another Variable.  Every Variable has a base Variable;
                a base Variable is its own base Variable.  A base variable is
                created when a Variable is initialized from a Tensor or when we
                call .detach().  The base Variable is the unit at which we make
                the decision "is this being tracked?".  As soon as a Variable
                becomes tracked, all Variables sharing the same base Variable
                become tracked.

      tick:   a tick is the value of a global 64-bit time counter that we increment every
              time we mutate a Tensor.  When we create Ops for backpropagation of
              derivatives, we record the tick at which the Op was created.
    invalidated:  if some data used in backprop needs to have been unchanged since
              a particular tick (as recorded in an Op), but it has been changed
              since then, we say that it has been invalided.  This is an error, but
              it will only be detected in debug mode.  In effect we store a
              record of what time (in ticks) data last changed at the individual-element
              level, via the ChangeTracker object that is attached to the Storage
              object (it's done in a structured way, not via a huge array).
              This means that the change-tracking mechanism is not defeated by
              doing .detach() or by constructing multiple Variables from the same
              Tensor.


*/


void Add(const Variable &a, const Variable &b, Variable *c) {
  // assumes c already correctly sized.


  Add(a.data(), b.data(), &(c->data()));

  Variable *a_grad = a->grad(), *b_grad = b->grad(),
      *c_grad = c->grad();

  auto gradFunc = [a_grad,b_grad,c_grad] () {
    a_grad->Add(*c_grad);
    b_grad->Add(*c_grad);
  }

  c->SetGradFunc(gradFunc);
  c->SetDependencies(a, b);

}


/*
  This is the 'gradient information' that class Variable stores
  when it is initialized with requires_grad = true (or is a result of
  an operation on Variables one of which had requires_grad = true).
  The Variable holds it via a shared_ptr.
  This does not give you access to the underlying Variable; doing it
  like this makes reference counting easier (no loops).  The GradFunc
  will store any pointers to the original Variable that it may have
  needed.

  Users will rarely need to interact directly with this struct.
 */
struct TensorGrad {
  // The version of the underlying Tensor.  (this number in the TensorGrad
  // mirrors that in the Variable; it's needed because TensorGrad's
  // 'inputs' variable refers back to the TensorGrad and does not have
  // access to the Variable).
  int32 version;


  struct InputInfo {
    int32 version;  // the version of the input that we used.  Used so we can
                    // check in the backprop that grad->version == version;
                    // if not, the user did something we don't allow.
    std::shared_ptr<TensorGrad> grad;
  };

  // The gradients corresponding to the input variables, which
  // we may need to update.  Some subset of these may be nullptr,
  // corresponding to input Variables for which no gradient
  // was required.
  std::vector<InputInfo> inputs;

  // is_view is true only if the Variable underlying this TensorGrad
  // is the result of an expression like foo.transpose() that creates
  // a view to another Tensor.  In that case, the variables
  // 'meta' and 'offset' become relevant, and when asked to create
  // the 'grad' Variable, we won't allocate it directly but will
  // instead create a view into inputs[0].grad->data.
  bool is_view{false};

  // grad_discarded will be set to true in the backprop when we are done
  // with this->grad and have deallocated it.  If a future user
  // attempts to reallocate the gradient, this will trigger an
  // exception.
  bool grad_discarded{false};

  // This contains the meta-information of the Tensor for which this is the
  // gradient (its 'data' pointer will be NULL).  Used to set up 'grad' with the
  // correct dimension and strides when it is needed.
  TensorMeta meta;
  // Only if is_view == true, the offset (in elements) of the start of
  // the Tensor described in 'meta' from the start of the source Tensor.
  // Used in constructing 'grad'
  int64 offset;

  // This stores the gradient (if we already have one), or nullptr if not.
  std::unique_ptr<Variable> data;

  // The tail in a singly linked list of TensorGrads... used in case this
  // Variable is a sum of several terms that were added together in-place.
  std::unique_ptr<TensorGrad> tail;

  // You call this function to ensure that the 'grad'
  void EnsureGradAllocated();
};



struct TensorGradOp {
  std::vector<std::shared_ptr<TensorGraph> > inputs;
  std::vector<std::shared_ptr<TensorGrad> > outputs;


  std::vector<std::shared_ptr<Variable> > vars_needed;

  std::function<void()> op;


  TensorGradOp(std::initializer_list<VariableRef> inputs_grads_needed,
               std::initializer_list<VariableRef> output_grads_needed,
               std::initializer_list<VariableRef> variables_needed,
               std::function<void()> op);

};

/**
   This contains the graph-related information stored with a Variable.
   For Variables initialized with requires_grad = true, it's held
   via shared_ptr as graph_.
 */
struct TensorGraph {
  // creator_ops contains the op that created (or modified) the Variable that
  // this TensorGraph is held by.  (If it modified this variable, 'tail' records
  // any previous operations on it).
  std::shared_ptr<TensorGradOp> creator_op;

  std::shared_ptr<TensorGrad> grad;

  std::shared_ptr<TensorGraph> tail;
};


/**
   GradFunc is the type that is passed into the constructor of Variable by a
   function implementing some operation on Variables (addition, multiplication,
   etc.).  It is at the core of the backprop mechanism, so we explain it here

 */
typedef std::function<void(const Variable &grad, const std::vector<Variable> *input_grads)> GradFunc;

typedef std::function<void(TensorGrad *grad)> GradHook;


// This is an enum but will be used as if it were an int32,
// as a bit pattern.
// TODO: figure out proper way to do that.
// It's used as an arg to the constructor of Variable
class enum VariableInit {
  CopyData = 1,
  CopyGrad = 2,
  CopyGraph = 4
};


// Shared data of a base Variable.  Each base Variable gets one of these; but
// non-base Variables (views into other variables) share the Node of their base
// Variable.
struct Node {
  // The gradient.
  Tensor grad_;


  // op_list_ (may be NULL) is the head of a list of Ops
  // that wrote to this Node (most recent at the head).
  // TODO: make it unique_ptr?
  std::shared_ptr<Op> op_list_;
};


class Variable;


/**
   class Variable is somewhat like class Tensor but augmented with autograd
   machinery.
*/
class Variable {


  /** Constructor from a Tensor.
       @param [in] data  Pointer to the source Tensor.  Will accept a
                      raw Tensor* pointer, in which case it will construct a
                      shared_ptr.  (??)
       @param [in] requires_grad    If requires_grad argument is true,
                the gradient w.r.t. this Variable will be computed if and when
                you call Backward() on a Variable that depends on it.
                The same as requires_grad in PyTorch.
  */
  Variable(const std::shared_ptr<Tensor> &data, bool requires_grad);



  /**  Returns shared pointer to the Tensor storing the data. */
  std::shared_ptr<Tensor> Data();


  /**  Returns pointer to the Tensor storing the derivative w.r.t.  this
       data.  Obtaining this Tensor won't allocate the memory, thanks to lazy
       initialization.  It is an error to call this if this Variable is
       not tracked (search for "Tracked:" above for definition).
       See also GradDataIfPresent().
  */
  std::shared_ptr<Tensor> GradData();

  /**  Returns pointer to the Tensor storing the derivative w.r.t.  this
       data, or NULL if not present..  Obtaining this Tensor won't allocate the
       memory, thanks to lazy initialization.  See also GradData().
  */
  std::shared_ptr<Tensor> GradDataIfPresent();


  /**
     Returns pointer to the base Variable (which may or may not be
     identical to 'this'.
   */
  std::shared_ptr<Variable> GetBaseVariable();

  /**
     Constructor that will be used by functions implementing mathematical
     operations on Variables.


     @param [in] data    Data to be stored in the Variable
     @param [in] inputs  A vector containing Variables which this Variable
                         depends on (for backpropagation purposes; will
                         be stored in the TensorGrad object).
     @param [in]

     a vector specifying inputs for this Variable
   * @param[in] gradFunc function specifying how to calculate gradient of the
   * input Variables
   */
  Variable(std::shared_ptr<Tensor> &data, std::vector<Variable> inputs,
           GradFunc grad_func);


  // Returns true
  bool Tracked() const;



 private:

  // data_ is the Tensor underlying this Variable.
  std::shared_ptr<Tensor> data_;

  // base_ is the base Variable which is non-NULL only if this Variable is a
  // view of an underlying Variable.  This needs to be tracked even if
  // we are not yet tracking gradients, because if any Variable with a
  // particular base becomes tracked, all such Variables do.
  // If base_ is NULL, then this Variable is its own base variable.
  std::shared_ptr<Variable> base_;

  // grad_ is a pointer to the struct containing gradient information (for
  // Variables that require a gradient; else NULL).  It may also be
  // NULL because someone called this->RemoveGrad().
  std::shared_ptr<TensorGrad> grad_;

  // ops_ is the first in singly list of Ops for this Variable.  If this
  // Variable is not its own base variable (i.e. if base_ != NULL), this will be
  // NULL since the Ops are only stored in the base Variables.


  (there will be just one element,
  // unless in-place operations were done).
  // Will be NULL if this Variable does not require a gradient or if someone
  // called this->RemoveGraph().
  std::shared_ptr<Op> ops_;
};

typedef std::unique_ptr<Storage>




};


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_VARIABLE_H_

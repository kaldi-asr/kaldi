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

#ifndef KALDI_TENSOR_VARIABLE_H_
#define KALDI_TENSOR_VARIABLE_H_ 1

#include "tensor/variable.h"

namespace kaldi {
namespace tensor {




// Shared data of a base Variable.  Each tracked base Variable gets one of
// these; non-base Variables (views into other variables) share the Node of
// their base Variable.
class Node {


  /**
     Construct a Node.
          @param [in] tensor   The data_ of the base Variable to which
                     this node is to be attached.  The created 'grad' will
                     have the same dims but different Storage and possibly
                     different strides;
  */
  explicit Node(const Tensor &tensor);



  /**
     This is to be used when setting the grad_ member of view variables.  it
     constructs a new Tensor with the appropriate pattern for the view, but
     pointing to the storage of 'grad'.
          @param [in] tensor  The data_ of the view Variable for which
                      we are requesting the gradient Tensor.
          @return     Returns a Tensor that is a view into 'grad', with
                      the same relationship to it as 'tensor' had to
                      its underlying Variable.
  */
  std::shared_ptr<Tensor> GetGradFor(const Tensor &tensor);


  /**
     Sets the most recent Op held here (latest_op).  This is called whenever
     an Op is created that changed a Variable attached to this Node.  The
     Op itself will have a shared_ptr to the previous Op that was attached
     to this Node.
   */
  inline void SetOp(const std::shared_ptr<Op> &op) { latest_op = op; }

  // The gradient.  This is set up when the Node is created, but the data in its
  // Storage object won't necessarily have been allocated (see "Lazy Allocation"
  // in tensor.h)
  std::shared_ptr<Tensor> grad;

  // Either NULL, or an object capable of converting patterns from
  // tensor to gradients (used for views).  Will be NULL in the usual
  // case where the Tensor for this base Variable has the same strides
  // and offset as the grad.
  std::unique_ptr<TensorPatternRebaser> rebaser;

  // latest_op is the most recent of the Ops that modified the base Variable
  // this is attached to, or any view into it.

  // op_list (will usually be NULL) is the head of a list of Ops that wrote to
  // this Node (the most recent first).  In the backward pass we call Backprop()
  // on each of these Ops in turn.  TODO: make it unique_ptr?
  std::shared_ptr<Op> latest_op;

 private:
  Node(const Node &other);  // Disallow copy construction
  Node & operator = (const Node &other);  // Disallow assignment
};


/**
   Implementation class for Variable.  Variable is just a shared_ptr to this.
 */
class VariableImpl {

  inline const std::shared_ptr<Tensor> &GetData() const { return data_; }


  // Returns true if this Variable is tracked (see "Tracked" in the
  // glossary in tensor.h).
  inline bool Tracked() const;


  // Returns the most recent Op in the autograd

  inline const std::shared_ptr<Op> &GetOp();

  // Returns the node in the autograd graph, as a shared_ptr; this creates it if
  // it did not exist (so the Variable, and others sharing the same base
  // Variable, will become tracked if it was not before).
  inline const std::shared_ptr<Node> &GetNode();

  // Returns the Tensor corresponding to the gradient; like GetNode, this will
  // make the Variable tracked if it were not tracked before.
  inline const std::shared_ptr<Tensor> &GetGrad();

 private:
  // Creates the node in the autograd graph.  This must be a base Variable
  // and the node must not already exist (i.e. we require node_ == NULL,
  // base_ == NULL).
  void CreateNode();

  // The Tensor that this Variable wraps.  Will always be non-NULL.  (Lazy
  // allocation may still happen in its Storage object, until we do something
  // with it).
  std::shared_ptr<Tensor> data_;

  // 'node_' is the node in the autograd graph, which is only allocated for
  // tracked base Variables; otherwise it is NULL.  It is allocated at the time
  // we realize we need gradient tracking, which might be when we create the
  // Variable, or later on if an in-place operation on it has as input a tracked
  // Variable.
  //
  // Non-base Variables cache the node of their base Variable, but if their node
  // is requested and this pointer is NULL and base_ is non-NULL, we need to
  // look at base_->node_ to re-check whether the base Variable is tracked, in
  // case it became tracked since we last checked.
  std::shared_ptr<Node> node_;

  // A pointer to the gradient, or NULL if this Variable is not tracked (Note:
  // like node_, this can get out of date if the base Variable becomes tracked,
  // so if base_ != NULL, we need to re-check).  grad_ can then be created
  // from the information in node_, once it exists; it is cached here
  // for efficiency.  See "Lazy Allocation" in glossary in tensor.h;
  // the underlying data may not have been allocated.
  // grad_ and node_ are always either both NULL or both non-NULL.
  std::shared_ptr<Tensor> grad_;


  // 'base_' is NULL if this is a base Variable (i.e. not a view of another
  // Variable); otherwise it points to the base Variable.  This also requires
  // that class Variable store its VariableImpl as a shared_ptr.
  std::shared_ptr<VariableImpl> base_;


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
     Returns the most recent Op that modified the base Variable of this
     Variable.  This will be called so the dependency can be recorded in other
     Ops, and also if called Backprop() and we want to create the list
     of Ops to do backprop on.
   */
  std::shared_ptr<Op> GetOp();


  /**
     Sets the most recent Op held in the Node underlying this Variable
     to the Op held in this shared_ptr (which must not be NULL).  This
     is done whenever we create an Op that modifies a particular
     Variable.
  */
  void SetOp(const std::shared_ptr<Op> &op);

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


p


 private:
  std::shared_ptr<VariableImpl> impl_;
};






};


}  // namespace tensor
}  // namespace kaldi


// Include implementation of inline functions.
#include "variable-inl.h"


#endif  // KALDI_TENSOR_VARIABLE_H_

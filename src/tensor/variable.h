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
     Sets the most recent Op held here (op_).  This is called whenever
     an Op is created that changed a Variable attached to this Node.  The
     Op itself should have a shared_ptr to the previous Op that was attached
     to this Node.
   */
  inline void SetOp(const std::shared_ptr<Op> &op) { op_ = op; }

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
   This is an overflow from class VariableImpl of various rarely-used fields; we
   instantiate it only when they are used.  This avoids bulking up the
   implementation of VariableImpl with them.


 */
struct VariableImplAux {

  // rebaser_ is always NULL for view Variables.   For tracked base
  // Variables where data_ and grad_ have different offset and/or
  // strides, it is an object capable of converting patterns from
  // tensors to gradients (used when constructing views).
  std::unique_ptr<TensorPatternRebaser> rebaser;

  // config_ is NULL if no config values have been stored; otherwise,
  // a pointer to class Config.
  std::unique_ptr<Config> config;


};


/**
   Implementation class for Variable.  Variable just holds a shared_ptr to this.
 */
class VariableImpl {
 public:

  inline const Tensor &GetData() const { return data_; }

  // Returns true if this Variable is tracked (see "Tracked" in the
  // glossary in tensor.h).
  inline bool Tracked() const;

  // Returns the most recent Op in the autograd graph (will return the same
  // value for all Variables sharing the same base Variable).  Will be
  // NULL if this Variable was not tracked.
  inline const std::shared_ptr<const Op> &GetOp() const;

  // Returns the Tensor corresponding to the gradient; this will make the
  // Variable (and any other Variable sharing the same base Variable) tracked if
  // it was not tracked before (see "Tracked" in glossary in tensor.h)
  inline const Tensor& GetGrad();


  // Returns the Tensor corresponding to the gradient if this variable is
  // tracked; else returns NULL; Differs from GetGrad() in its behavior for
  // non-tracked Variables.
  inline const std::shared_ptr<const TensorImpl>& GetGradIfTracked();


  // Sets the most recent Op for the base Variable of this Variable;
  // this is called by Ops to register themselves with Variables that
  // they modify.
  inline void SetOp(std::shared_ptr<const Op> &op);


  // This function must only be called on tracked base Variables (see glossary
  // in tensor.h; it requires grad_ != NULL and base_ == NULL).  It gets the
  // grad Tensor corresponding to the data in 'data', which is assumed to
  // be a view into this->data_.  This grad Tensor will be a view into
  // this->grad_.  This function is called from view Variables when setting
  // up their grad_ variables.
  inline Tensor GetGradForView(const Tensor &data);

 private:

  // This function, which must only be called on a non-tracked base Variable,
  // creates the 'grad_' tensor.
  void CreateGrad();

  // The Tensor that this Variable wraps.  (Note: it just holds a non-NULL
  // shared_ptr<const TensorImpl>.  The const is to ensure the meta-info
  // isn't changed unexpectedly).
  Tensor data_;

  // The gradient corresponding to `data_`, or NULL if this is:
  //  (a) a base Variable that is not tracked, or
  //  (b) a view Variable that is either not tracked, or we have
  //      not yet cached the gradient.  (we might need to follow
  //      the base_ pointer to get the gradient).
  // Note: the data underlying this gradient is not necessarily allocated; see
  // "Lazy Allocation" in the glossary in tensor.h.
  // The type differs from Tensor only because it might be NULL.
  std::shared_ptr<const TensorImpl> grad_;

  // 'base_' is NULL if this is a base Variable (i.e. not a view of another
  // Variable); otherwise it points to the base Variable.
  std::shared_ptr<VariableImpl> base_;

  // op_ is always NULL for view Variables.  For tracked base Variables,
  // it is the most recent Op that modified this Variable.  (The autograd
  // graph is solely between Ops; this latest_ is our entry point to that
  // graph and is also used in its construction).
  std::shared_ptr<const Op> op_;

  // For tracked base Variables, this will be set to true if the pattern of
  // grad_ is different from the pattern of data_ (because data_ was not
  // contiguous and justified), and false otherwise.  If this is true, we need
  // to rebase any views of this variable.  For non-tracked or non-base
  // Variables, its value is undefined.
  bool rebase_grad_;

  // overwrite_ is part of a mechanism that avoids unnecessary zeroing of
  // parts of derivatives during the backprop phase.  By default we
  // assume that if we write to a Variable in a way that doesn't
  // depend on the previous value (e.g. we set it, rather than
  // add to it or multiply in-place), then the previous memory underlying
  // that Variable has not previously participated in any operations
  // requiring derivatives.
  //
  // If you are about
  // to modify a Variable c that *has* previously participated in
  // operations requiring derivatives, then, instead of, say:
  //  DoSomethingWith(a, b, &c);
  // (and let's suppose this operation ignores the previous value of `c`),
  // you could do:
  //  DoSomethingWith(a, b, &c.Overwrite());
  // whereby you assert that the memory underlying this variable may have
  // previously participated in operations requiring derivative tracking
  // (and hence we need to an extra zeroing after the backprop).
  // The call to Overwrite() sets the `overwrite_` bool, and then
  // the DoSomethingWith() call should unset it.
  //
  // Look at the comment for class InvalidatedDataChecker in change-tracker.h
  // for more information.
  bool overwrite_;

  // aux_ is basically a collection of less-often-used fields of class VariableImpl;
  // it helps keep the main class uncluttered.
  std::unique_ptr<VariableImplAux> aux_;
};


class Variable;


/**
   class Variable is like class Tensor but augmented with autograd machinery.
*/
class Variable {

  /** Constructor from a Tensor.
       @param [in] data  The source Tensor.  (This Variable will copy it; this
                      is to avoid errors if you change the original Tensor).

       @param [in] requires_grad    If requires_grad argument is true,
                the gradient w.r.t. this Variable will be computed if and when
                you call Backward() on a Variable that depends on it.
                The same as requires_grad in PyTorch.
  */
  Variable(const Tensor &data, bool requires_grad);



  /**  Returns shared pointer to the Tensor storing the data. */
  const Tensor &Data() const;


  Tensor &Data();


  /**  Returns pointer to the Tensor storing the derivative w.r.t.  this
       data.  Obtaining this Tensor won't allocate the memory, thanks to lazy
       initialization.  Calling this will make this Variable tracked.
  */
  Tensor &GradData();


  /**  Returns pointer to the Tensor storing the derivative w.r.t.  this data if
       this Variable is already tracked, or NULL if not.  This is for framework
       use, not for users.  Note: shared_ptr<TensorImpl> means "maybe a Tensor,
       maybe NULL".
  */
  std::shared_ptr<const TensorImpl> GradDataIfPresent();


  /**
     Returns pointer to the base Variable (which may or may not be
     identical to 'this'.
   */
  Variable GetBaseVariable();


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




 private:
  // You may ask: Variable is just a shared_ptr<VariableImpl>, so why not just
  // get rid of it, rename VariableImpl to Variable, and give people the choice
  // of what memory management approach to use?  The issue is, we *require* the
  // use of shared_ptr because the `base_` pointer in VariableImpl is also a
  // shared_ptr to VariableImpl.  Forcing the users to always supply a
  // shared_ptr<Variable> seems like a bad pattern, so we use this `impl_`
  // approach where the shared_ptr is hidden.  This is similar to class Tensor,
  // although the VariableImpl is not const because (for instance) we may
  // need to make it tracked if it isn't currently.
  //
  // The difference between a Variable and std::shared_ptr<VariableImpl> is that
  // the latter may be NULL, but a Variable never has a NULL impl_.
  std::shared_ptr<VariableImpl> impl_;
};






};


}  // namespace tensor
}  // namespace kaldi


// Include implementation of inline functions.
#include "tensor/variable-inl.h"


#endif  // KALDI_TENSOR_VARIABLE_H_

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


enum OpProperties {
  kConcreteOp = 1,  // An Op that is concrete is one that can be executed
                    // directly, i.e. its Do() function works; these Ops will
                    // generally correspond to a single function call, e.g. a
                    // particular BLAS call If an Op is not concrete, you should
                    // keep expanding via Expand() until you get concrete ops,
                    // and then execute those.
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
     if that was what this Op did).  Only needs to be defined for Ops that
     are concrete, i.e. Properties() & kOpConcrete
  */
  virtual void Do() const {
    KALDI_ERR << "Execution not supported for this Op (not concrete); "
        "please expand ";
  }

  /**
     Return a copy of this object, newly allocated using new.
  */
  virtual Op *Copy() const = 0;


  /**
     Properties of this Op, a bunch of boolean flags such as kConcreteOp
     (may add more in future)
   */
  virtual int32 Properties() const = 0;

  /**
     To be called only for non-concrete Ops, i.e. Ops for which Properties() &
     kConcreteOp is zero.  Calling this function will expand this Op into one or
     more concrete Ops, appending them to 'ops'.
        @param [out] ops
                     Operations will be *appended* to `ops`.  These operations
                     will be fully-expanded versions of this Op.  (i.e. they
                     will be concrete).
   */
  virtual void Expand(std::vector<std::unique_ptr<Op> > *ops) = 0;


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
                                  std::vector<std::unique_ptr<Op> > *ops) const {
    KALDI_ERR << "Forward-mode autograd not supported for this Op";
  }


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
                                   std::vector<std::unique_ptr<Op> > *ops) const {
    KALDI_ERR << "Reverse-mode autograd not supported for this Op";
  }



  /** Destructor.  It's important for efficiency of memory use to destroy Ops as
      soon as you won't need them any more, because it may trigger the freeing
      of Tensors and hence Storage regions.
  */
  virtual ~Op();
 protected:

  // This function ensures that the *last element* of `ops` is fully expanded At
  // entry, `ops` is a nonempty vector of Op pointers, which are all concrete
  // except the last entry.  At exit, `ops` is a nonempty vector of Op pointers
  // which are all concrete.  This function will usually be called from Expand()
  // after code that appends an Op that might not be concrete to `ops`.
  void EnsureExpanded(std::vector<std::unique_ptr<Op> > *ops) {
    if (!(ops->back()->Properties() & kConcreteOp)) {
      Op *op = ops->back().get();
      ops->pop_back();
      op->Expand(ops);
    }
  }

};

// See linear-ops.h and nonlinear-ops.h for concrete examples of Ops.

}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_VARIABLE_H_

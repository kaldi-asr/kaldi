// tensor/tensor-impl-linear.h

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

#ifndef KALDI_TENSOR_IMPL_LINEAR_H_
#define KALDI_TENSOR_IMPL_LINEAR_H_ 1

#include "tensor/tensor.h"


/**
   This header contains basic linear-algebra and copying types of operations
   on TensorImpl objects.  See also tensor-impl-nonlinearly
 */

namespace kaldi {
namespace tensor {

/**
   Modifies 't' in-place by inserting an axis with (dim=1,stride=0) at the
   specified position.  Updates the code.

   A negative axis-index i is interpreted (like PyTorch) as (num_axes + 1 - i).

   Showing just the dims in the tensor for some examples:

\verbatim
    Unsqueeze({3,4}, 0)  -> {1,3,4}
    Unsqueeze({3,4}, 1)  -> {3,1,4}
    Unsqueeze({3,4}, 2)  -> {3,4,1}
    Unsqueeze({3,4}, -1)  -> {3,4,1}
    Unsqueeze({3,4}, -2)  -> {3,1,4}
\endverbatim
 */
void Unsqueeze(TensorImpl *t, int32 axis)


/**
   Modifies 't' in-place by removing an axis with (dim=1,stride=0) from the
   specified position.  It is an error if 't' did not initially contain
   such an axis.

   Showing just the dims in the tensor for an example:

\verbatim
    Unsqueeze({1,3,4}, 0)  -> {3,4}
    Unsqueeze({3,1,4}, 1)  -> {3,4}
    Unsqueeze({3,1,4}, 2)  -> [error]
\endverbatim
 */
void Squeeze(TensorImpl *t, int32 axis);



/**
   Does:

    `c := alpha (a * b)  +  beta c`

   where '*' is elementwise multiplication subject to broadcasting rules.  This
   version supports reducing and broadcasting operations, and is where
   matrix multiplication actually gets implemented; see Matmul().

   The Tensors do not all have to have the same NumAxes(); they will
   (conceptually) be made the same size by padding on the left with trivial axes
   (dim=1;stride=0) to make them the same size.

   The Tensors need to have the same Dtype() and Device().

   @param [in] alpha  Value that scales a * b
   @param [in] beta   Value that scales the initial value of c
   @param [in] a      First input tensor
   @param [in] b      Second input tensor.
   @param [out] c     Tensor to be added to; we require Broadcastable(a, b, c).
                      and either c's data must be initialized to a known
                      value (if beta != 0) or known to not contain NaN (if
                      beta == 0); but we have to figure out whether we can drop
                      the NaN requirements as some BLAS's may treat beta=0
                      specially.
 */
void AddProduct(float alpha, float beta,
                const TensorImpl &a, const TensorImpl &b,
                const TensorImpl *c);


/**
   Copy elements from Tensor a to Tensor b, possibly broadcasting
      @param [in]  a    The source Tensor.
      @param [out] b   The destination Tensor.  We require
                       Broadcastable(a, b, true).

   See also Add(), which is more general than Copy.
 */
void Copy(const TensorImpl &a, const TensorImpl *b);


/**
   Add elements from Tensor a to Tensor b, broadcasting or summing
   as dictated by the dimensions involved; does
      \f$  b := \alpha a + \beta b.  \f$

      @param [in]  a    The source Tensor.
      @param [out] b   The destination Tensor.  We require
                       Broadcastable(a, b).
 */
void Add(float alpha, float beta,
         const TensorImpl &a, const TensorImpl *b);


/**
   Matrix multiplication; does

     \f$  c := \alpha a b   +  \beta c  \f$

   where `a b` is interpreted as matrix multiplication.  This generalizes in the
   same way as PyTorch's matmul does if there are extra dimensions in the args.
   In fact it generalizes more than that, encompassing cases where the matrix
   product may be summed over certain dimensions.

   The implementation is just:

     Tensor a_tmp(a), c_tmp(c);
     a_tmp.Unsqueeze(-1);
     c_tmp.Unsqueeze(-2);
     AddProduct(alpha, beta, a_tmp, b, c_tmp);

 */
void Matmul(float alpha, float beta,
            const TensorImpl &a, const TensorImpl &b,
            const TensorImpl *c);



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_IMPL_LINEAR_H_

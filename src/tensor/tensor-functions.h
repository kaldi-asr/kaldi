// tensor/tensor-functions.h

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

#ifndef KALDI_TENSOR_FUNCTIONS_H_
#define KALDI_TENSOR_FUNCTIONS_H_ 1

#include "tensor/tensor.h"

namespace kaldi {
namespace tensor {

// This file contains functions that operate on Tensors in various ways.  To
// avoid class Tensor blowing up hugely, we implement these things outside
// class Tensor.


// Note: we use the distinction between references and pointers the same way as
// you might expect from Google-style-guide code, to reflect which Tensors'
// contents are changed (so a pointer argument might have its contents changed.
// But these are in most cases pointers to const Tensors; they can be
// donst because the metadata is not changed, even if the data is.


// Sets all elements of the tensor to zero.
void SetZero(const Tensor *tensor);

// Sets all elements of the tensor to value f (cast to whatever type
// this Tensor has).
void SetZero(float f, const Tensor *tensor);


// Return a transposed version of this Tensor that shares the underlying memory.
Tensor Transpose(const Tensor &tensor, int64_t axis1 = 0, int64_t axis2 = 1);

/**
   Copy the data from tensor 'src' to tensor 'dest'.  Does not change the tensor
   metadata, but does change the data underlying the Tensor 'dest'.

   Requires that src.Dims() an dest->Dims() be compatible, meaning that they are
   the same, except it's OK for a dim of 'dest' to be 1 and a dim of 'src' to be
   >1; in such cases, we will broadcast the element from 'src' across the larger
   dimension of 'dest'.

   Does not require that the Dtype() or Device() of src and dest be the
   same.
*/
void CopyData(const Tensor &src, const Tensor *dest);

/**
  Construct, if possible, a Tensor that is a view into 'src' with the
  requested dimensions.

  The semantics are based on those of PyTorch's "view" or NumPy's
  "reshape", except we try to be more agnostic about the striding
  of the input.

  Consider a Tensor 'a' has "C"-style strides.  Then this function will return
  Tensor (say, 'b') that interprets the raw data of 'a' as an array with
  "C"-style strides but with dimensions 'dims'.  (The product of 'dims' must
  equal src.NumElements()).

  Now consider a Tensor 'a2' that does not have "C"-style strides but
  has the same elements as 'a' in the sense that a(i,j,k) == a2(i,j,k).
  Then, *if possible*, this function will return a matrix b2 with
  the same elements as b, e.g. b2(i,j,k) == b(i,j,k).

  This function returns NULL if such a tensor could not be constructed.  In that
  case, likely what you will want to do is to construct a temporary Tensor from
  'src' with the same dimensions but "C"-style strides (see the constructor of
  Tensor that accepts the 'dims' parameter).  You may then call View() on that
  temporary Tensor, which is guaranteed to succeed.

     @param   [in] src   The source Tensor, whose data is to be
                         reinterpreted.
     @param   [in] dims  The dimensions that we want for the returned
                       Tensor; its product must equal src.NumElements().
     @param   [out] dest  If the view could be constructed, this function
               make 'dest' a view of the data in 'src' with the requested dims;
               otherwise 'dest' will be unchanged.
     @return   Returns true if this view could be constructed. If
               src.HasCStrides() is true, this function will never return
               false.
 */
bool View(const Tensor &src, ArrayRef<int64_t> dims, Tensor *dest);


/**
   Returns a Tensor with a new view of the data in 'src', in which the axes
   numbered axis1 and axis1 + 1 are merged.  This is just a special case
   of View().

   For example, if 'src' is a Tensor with dims (3,4,5) and you call
   MergeAxes(src, 1), this funtion will merge axes 1 and 2 and return a Tensor
   with shape (3,20).  The order of the elements in the second axis of the
   result is required to be what you would expect if the layout as as a
   "C" array (so: 4 blocks of 5 elements, and not vice versa).  This
   is a common special case of what the function 'View' can give you.

   If the pattern of 'src' makes the requested merging impossible,
   this function will return NULL.  (This will happen if, in the
   Tensor 'src', stride[axis1+1] != stride[axis1] * dim[axis1]).

   If this function returns NULL then the caller will probably want to construct
   a temporary Tensor 'temp' passing src.Dims() in the constructor, copy the
   data in 'src' to 'temp', and then call MergeAxes on 'temp'.

       @param [in]  src   The Tensor whose axes we will attempt to
                          merge
       @param [in] axis1  The index of the first of the two axes which
                          this function will attempt to merge.  Must
                          be less than src.NumAxes() - 1.
       @param [out] dest  The Tensor which is written to; on success this
                          will be a Tensor with axes merged as requested,
                          sharing the data of 'src'.  On failure, it will
                          not be changed.
       @return            Returns true on success, false if the axes could
                          not be merged (e.g., because of the strides not
                          having the required relationship).
 */
bool MergeAxes(const Tensor &src, int64_t axis1, Tensor *dest);

/**
   Creates a Tensor in which the axis numbered 'axis' is split into
   two axes, with dimensions respectively 'dim1' and 'dim2'.  The
   interpretation will be as for a "C" array; so, for instance,
   if the dimensions of 'src' were (10,12) and you called
   `SplitAxis(src, 1, 3, 4)` resulting in a Tensor of dimensions
   (10,3,4), the indexes along the original axis of dimension 12 would be
   interpreted as 3 blocks of size 4.  (This is the normal semantics
   of things like NumPy's reshape or PyTorch's view.)

      @param [in] src  The Tensor whose axis is to be split.
      @param [in] axis  The index of the axis to be split; must
                       satisfy `0 <= axis < src.Dims().`
      @param [in] dim1  First dimension with which to split the axis.
      @param [in] dim2  Second dimension with which to split the axis.
                        Must satisfy `dim1 * dim2 == src.Dim(axis)`.
      @param [out] dest Tensor to be created, with one more axis than 'src',
                        sharing the same underlying data.
*/
void SplitAxis(const Tensor &src, int64_t axis,
               int64_t dim1, int64_t dim2,
               Tensor *dest);




/**
   Does:

    `c := alpha (a * b)  +  beta c`

   where '*' is elementwise multiplication subject to broadcasting
   rules.  This does not support reducing operations (see AddProductReducing).

   @param [in] alpha  Value that scales a * b
   @param [in] beta   Value that scales the initial value of c
   @param [in] a      First input tensor
   @param [in] b      Second input tensor; require BroadcastCompatible(a, b)
   @param [out] c     Tensor to be added to (must already be correctly sized,
                      and either its data must be initialized to a known
                      value (if beta != 0) or known to not contain NaN (if
                      beta == 0).   We require BroadcastCompatible(a, b, c, true).
 */
void AddProduct(float alpha, float beta,
                const Tensor &a, const Tensor &b, Tensor *c);



/**
   Does:

    `c := alpha (a * b)  +  beta c`

   where '*' is elementwise multiplication subject to broadcasting
   rules.  This version supports reducing operations (i.e. it allows
   'c' to have dim=1 on axes where a and/or b has dim!=1).

   This function actually supports a strict superset of AddProduct(); we
   separate the functions to make the implementation for AddProduct() simpler,
   for speed.

   The Tensors do not all have to have the same NumAxes(); they will
   (internally) be made the same size by padding on the left with trivial axes
   (dim=1;stride=0) to make them the same size.

   The Tensors need to have the same Dtype() and Device*().

   @param [in] alpha  Value that scales a * b
   @param [in] beta   Value that scales the initial value of c
   @param [in] a      First input tensor
   @param [in] b      Second input tensor; require BroadcastCompatible(a, b)
   @param [out] c     Tensor to be added to (must already be correctly sized,
                      and either its data must be initialized to a known
                      value (if beta != 0) or known to not contain NaN (if
                      beta == 0).   We require BroadcastCompatible(a, b, c).
 */
void AddProductReducing(float alpha, float beta,
                        const SubTensor &a, const SubTensor &b,
                        SubTensor *c);



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_FUNCTIONS_H_

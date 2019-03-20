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
     @return   If the view could be constructed, this function returns
               a shared_ptr to a new Tensor with the requestd dims,
               that shares underlying data with 'src'; otherwise returns
               NULL.  (If src.HasCStrides(), then this function is
               guaranteed not to return nullptr).

 */
std::shared_ptr<Tensor> View(const Tensor &src, ArrayRef<int64_t> dims);


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

       @param [in]  src   The Tensor which whose axes we will attempt to
                          merge
       @param [in] axis1  The index of the first of the two axes which
                          this function will attempt to merge.  Must
                          be less than src.NumAxes() - 1.
       @return            Returns a pointer to a Tensor with the
                          merged axes (if the pattern of 'src'
                          allows it), or nullptr otherwise.
 */
std::shared_ptr<Tensor> MergeAxes(const Tensor &src, int64_t axis1);

/**
   Returns a Tensor with a new view of the data in 'src', in which the
   specified axis is split into two axes.  This is just a special case
   of View().

   Returns a Tensor in which the axis numbered 'axis' is split into
   two axes, with dimensions respectively 'dim1' and 'dim2'.  The
   interpretation will be as for a "C" array; so, for instance,
   if the dimensions of 'src' were (10,12) and you called
   `SplitAxis(src, 1, 3, 4)` resulting in a Tensor of dimensions
   (10,3,4), the indexes along the original axis of dimension 12 would be
   interpreted as 3 blocks of size 4.

      @param [in] src  The Tensor whose axis is to be split.
      @param [in] axis  The index of the axis to be split; must
                       satisfy `0 <= axis < src.Dims().`
      @param [in] dim1, dim2   The two dimensions into which
                       we will split the axis.  Must satisfy
                       `dim1 * dim2 == src.Dim(axis)`.
      @return     Returns a Tensor which shares the same
                  underlying data as 'src'
 */
std::shared_ptr<Tensor> SplitAxis(const Tensor &src, int64_t axis,
                                  int64_t dim1, int64_t dim2);


}
}

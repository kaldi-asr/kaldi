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
   Construct, if possible, a Tensor that is a view into 'src'.  Read this
   carefully, as the semantics may differ from the 'view' functions in some
   other toolkits.  'View' does not care how the underlying data of 'src'
   is organized.  Its semantics can be explained as follows.

   Suppose `src` were a "C"-style array with dimension given by `src.Dims()`.
   Then reinterpret that array as one with dimension `dims`, if possible, and
   return a Tensor describing that array.  If



     @param

 */
std::shared_ptr<Tensor> View(const Tensor &src, ArrayRef<int64_t> dims);


}
}

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


}
}

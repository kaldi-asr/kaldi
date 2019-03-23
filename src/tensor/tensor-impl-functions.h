#include "tensor/tensor.h"
#include "tensor/tensor.h"


namespace kaldi {
namespace tensor {




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
                const TensorImpl &a, const TensorImpl &b, const TensorImpl *c);



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
                        const TensorImpl &a, const TensorImpl &b,
                        TensorImpl *c);



}
}

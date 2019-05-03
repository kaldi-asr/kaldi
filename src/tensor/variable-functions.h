// tensor/variable-functions.h

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

#ifndef KALDI_VARIABLE_FUNCTIONS_H_
#define KALDI_VARIALBE_FUNCTIONS_H_ 1

#include "tensor/tensor.h"

namespace kaldi {
namespace tensor {

// This file contains functions operating on Variables, mostly functions that
// return other Variables.


/**
   Return a Variable with all-zero values, with the specified dimensions

       @param [in] dims   Dimensions (in public ordering) of the requested
                      Tensor.  Must all be positive, with the length of
                      the list not exceeding KALDI_TENSOR_MAX_DIM = 6

  An example is below.
<code>
   Variable scalar = Zeros({});
   Variable a = Zeros({3,4}, {kDoubleDtype});
   Variable b = Zeros({1,100}, {kDoubleDtype, kGpuDevice});
</code>
  Note on C++: reading the code above may require getting used to C++
  braced-initializer-lists.  The {3,4} is interpreted as a
  std::inititializer_list<int32> passed to to the constructor of ArrayRef; the
  {kDoubleDtype} is an arg to the constructor of TensorOptions.
 */
inline Variable Zeros(ArrayRef<int32> dims,
                      TensorOptions opts = TensorOptions());


Variable Ones(ArrayRef<int32> dims);


/**
   Return a Tensor with
 */
Variable RandUniform(ArrayRef<int32> dims);

/**
   Sum all axes of a Variable and returns a Variable with one element and no
   axes.

       @param [in]  v   Variable to be summed.
       @return          The summation; will equal the sum over, all
                        axes of v; will have zero axes, and the same
                        device and dtype of 'v'.
Example:
<code>
   Variable v = Rand({3,4,5});
   Variable w = v.Sum();
</code>
   See also the version of Sum() for which you can specify axes.
 */
Variable Sum(const Variable &v);

/**
   Sum specified axes of a Variable.  The returned Variable will have
   that many fewer axes.

       @param [in] v      Variable to be summed
       @param [in] eaxes
 */
Variable Sum(const Variable &v, ArrayRef<int32> eaxes);





}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_FUNCTIONS_H_

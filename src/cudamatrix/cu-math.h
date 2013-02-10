// cudamatrix/cu-math.h

// Copyright 2009-2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CUMATH_H_
#define KALDI_CUDAMATRIX_CUMATH_H_
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-device.h"
#include "util/timer.h"

namespace kaldi {
  
namespace cu {
 
/// RegularizeL1 is a gradient step with l1 regularization added to the
/// gradient.  We don't let the value cross over zero from positive to negative
/// or vice versa, in a single step.  If an element tries to cross zero and is
/// stopped, we zero the gradient.  (Dan: not sure why).
template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *gradient,
                  Real l1_penalty, Real learning_rate);

/// ie. switch rows according to copy_from_idx
template<typename Real>
void Randomize(const CuMatrixBase<Real> &src,
               const CuStlVector<int32> &copy_from_idx,
               CuMatrixBase<Real> *tgt);

/// ie. concatenate the frames with offsets from frame_offsets
template<typename Real>
void Splice(const CuMatrix<Real> &src,
            const CuStlVector<int32> &frame_offsets,
            CuMatrix<Real> *tgt);

template<typename Real>
void Copy(const CuMatrix<Real> &src,
          const CuStlVector<int32> &copy_from_indices,
          CuMatrix<Real> *tgt);


} // namespace cu
} // namespace kaldi


#endif

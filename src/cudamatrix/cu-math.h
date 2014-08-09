// cudamatrix/cu-math.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (Author: David Snyder) 

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



#ifndef KALDI_CUDAMATRIX_CU_MATH_H_
#define KALDI_CUDAMATRIX_CU_MATH_H_
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-array.h"
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

/// Copies a permutation of src into tgt. The row permutation is specified in
/// copy_from_idx such that src.Row(copy_from_idx[r]) == tgt.Row(r). The 
/// dimensions of copy_from_idx must be equivalent to the number of rows in
/// tgt and src and all elements in the vector must be in [0, src.numRows()-1].  
template<typename Real>
void Randomize(const CuMatrixBase<Real> &src,
               const CuArray<int32> &copy_from_idx,
               CuMatrixBase<Real> *tgt);

/// Splice concatenates frames of src as specified in frame_offsets into tgt.
/// The dimensions of tgt must be equivalent to the number of rows in src
/// and it must be that tgt.NumColumns == src.NumColumns * frame_offsets.Dim().
/// As a result, tgt(i, k*n_cols + j) == src(i + frame_offsets[k], j) for the
/// general case where i in [0..src.NumRows()-1], 
/// k in [0..frame_offsets.Dim()-1], j in [0..src.NumRows()-1] 
/// and n_cols = src.NumColumns(). If i + frame_offsets[k] is greater than the
/// number of rows in src or less than 0 than the right side of the equation 
/// is replaced by src(src.NumRows()-1, j) or src(0, j) respectively, to avoid
/// an index out of bounds.
template<typename Real>
void Splice(const CuMatrixBase<Real> &src,
            const CuArray<int32> &frame_offsets,
            CuMatrixBase<Real> *tgt);

/// Copies elements from src into tgt as given by copy_from_indices.
/// The matrices src and tgt must have the same dimensions and
/// the dimension of copy_from_indices must equal the number of columns
/// in the src matrix. As a result, tgt(i, j) == src(i, copy_from_indices[j]).
/// Also see CuMatrix::CopyCols(), which is more general.
template<typename Real>
void Copy(const CuMatrixBase<Real> &src,
          const CuArray<int32> &copy_from_indices,
          CuMatrixBase<Real> *tgt);


} // namespace cu
} // namespace kaldi


#endif

// xvector/xvector.h

// Copyright 2016    Johns Hopkins University (Author: Daniel Povey)
//           2016    David Snyder

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

#ifndef KALDI_XVECTOR_XVECTOR_H_
#define KALDI_XVECTOR_XVECTOR_H_

#include <vector>
#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

namespace kaldi {
  /*
  Computes the training objective function and the derivatives for
  the xvector.  Let N = xvector_pairs.NumRows() be the number of
  xvectors. There are N(N-1)/2 pairs in total and N/2 from the same
  class. Let v(n) be the n'th row of the matrix xvector_pairs.
  The total objective function written to 'tot_objf' is
      \sum_{n=0}^{N/2} p_same(v(n*2), v(n*2+1))
      + 1/(N-2) \sum_{n=0}^{N} \sum_{m=2*ceil(n+1)/2)}^{N}
      p_different(v(m), v(n))
  and let N be the normalizer for the objective function, written to
  'tot_weight' and equal to the total (weighted) number of samples over
  which the objective function is computed. It is useful for displaying
  the objective function correctly.
  Let the log-odds L(v,w) [interpreted as log(p_same(v,w) / p_different(v,w))]
  be defined as:
      L(v, w) = v' w -  v' S v - w' S w + b
  then p_same(v, w) = -log(1 + exp(-l(v, w)), and
  p_different(v, w) = 1 - p_same(v, w) = -log(1 + exp(-l(v, w)).

  @param [in] xvector_pairs   Each row of 'xvector_pairs' is an xvector
  extracted by the network for one sample, and the assumption is that
  pairs of the form (2*k, 2*k+1), e.g., (0, 1), (2, 3), (4, 5), etc,
  are from the same class, but any other pairs, e.g., (0, 2), (1, 2),
  (2, 4), etc, are from different classes.
  @param [out] deriv_xvector  If non-NULL, the derivative of the objective
  function with respect to the xvectors is written here.
  @param [out] deriv_S  If non-NULL, the derivative of the objective
  function with respect to the parameter S are written here.
  @param [out] deriv_b  If other derivates are non-NULL the derivative of
  the objective function with respect to the parameter b is written here.
  @param [out] tot_objf  The total objective function described above
  @param [out] tot_weight  The total normalizing factor for the objective
  function, equal to xvector_pairs.NumRows().
  */
  void ComputeXvectorObjfAndDeriv(const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b,
    CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S,
    BaseFloat *deriv_b,
    CuMatrixBase<BaseFloat> *scores_out,
    BaseFloat *tot_objf,
    BaseFloat *tot_weight);

 /*
 Compute the similarity score between two input xvectors. The score is
 defined as:
      L(v, w) = v' w -  v' S v - w' S w + b
 @param [in] v  The first xvector.
 @param [in] w  The second xvector.
 @param [in] S  A symmetric matrix, usually a constant output of the
 Nnet the xvectors came from.
 @param [in] b  A scalar offset, usually a constant output of the Nnet
 the xvectors came from.
 @return  The score between vectors v and w.
 */
 BaseFloat SimilarityScore(const Vector<BaseFloat> &v,
    const Vector<BaseFloat> &w, const SpMatrix<BaseFloat> &S,
    BaseFloat b);

}  // namespace kaldi

#endif

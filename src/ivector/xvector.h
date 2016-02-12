// ivector/xvector.h

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

#ifndef KALDI_IVECTOR_XVECTOR_H_
#define KALDI_IVECTOR_XVECTOR_H_

#include <vector>
#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"

namespace kaldi {
  /*
  Computes the training objective function and the derivatives for
  the xvector.  Let N = xvector_pairs.NumRows() be the number of
  xvectors. There are N(N-1)/2 pairs in total and N from the same
  class. Let v(n) be the n'th row of the matrix xvector_pairs.
  The total objective function written to 'tot_objf' is
      \sum_{n=0}^{N/2} p_same(v(n*2), v(n*2+1))
      + 1/(N-2) \sum_{n=0}^{N} \sum_{m=2*ceil(n+1)/2)}^{N}
      p_different(v(m), v(n))
  and let 2*N be the normalizer for the objective function, written to
  'tot_weight' and equal to the total (weighted) number of samples over
  which the objective function is computed. It is useful for displaying
  the objective function correctly.
  Let the log-odds L(v,w) [interpreted as log(p_same(v,w) / p_different(v,w))]
  be defined as:
      L(v, w) = v' w -  v' S v - w' S w
  then p_same(v, w) = -log(1 + exp(-l(v, w)), and
  p_different(v, w) = 1 - p_same(v, w) = -log(1 + exp(-l(v, w)).

  @param [in] xvector_pairs   Each row of 'xvector_pairs' is an xvector
  extracted by the network for one sample, and the assumption is that
  pairs of the form (2*k, 2*k+1), e.g., (0, 1), (2, 3), (4, 5), etc,
  are from the same class, but any other pairs, e.g., (0, 2), (1, 2),
  (2, 4), etc, are from different classes.
  @param [out] deriv_xvector  If non-NULL, the derivative of the objective
  function with respect to the xvectors is written here.
  @param [out] deriv_S_and_b  If non-NULL, the derivative of the objective
  function with respect to the parameters S and b are written here.
  @param [out] tot_objf  The total objective function described above
  @param [out] tot_weight  The total normalizing factor for the objective
  function, equal to dvector_pairs.NumRows().
  */
  void ComputeXvectorObjfAndDeriv(const CuMatrixBase<BaseFloat> &dvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b,
    CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S_and_b,
    BaseFloat *tot_objf,
    BaseFloat *tot_weight);

  /*
  Computes a similarity score between xvectors v and w. It is defined as:
      L(v, w) = v' w -  v' S v - w' S w + b
  @param [in] v  The first xvector in the pair.
  @param [in] w  The second xvector in the pair.
  @param [in] S  A symmetric matrix used in the similarity score.
  @param [in] b  A scalar offset.
  @return  The similarity score between v and w.
  */
  BaseFloat SimilarityScore(const CuVector<BaseFloat> &v,
                            const CuVector<BaseFloat> &w,
                            const CuSpMatrix<BaseFloat> &S, BaseFloat b);
  /*
  Gets the derivative for a term in the objective function summation.
  @param [in] v  The first xvector in the pair.
  @param [in] w  The second xvector in the pair.
  @param [in] S  A symmetric matrix used in the similarity score.
  @param [in] b  A scalar offset.
  @param [in] is_same  Whether or not v and w are from the same or different
  classes.
  @param [in] similarity_score  The similarity score between v and w.
  @param [out] deriv_v  The derivative with respect to v.
  @param [out] deriv_w  The derivative with respect to w.
  @param [out] deriv_S_and_b  The derivative with respect to S
  (serialized), concatenated with the derivative with respect to b.
  */
  void GetDeriv(const CuVector<BaseFloat> &v,
    const CuVector<BaseFloat> &w, const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, bool is_same, BaseFloat similarity_score,
    CuVector<BaseFloat> *deriv_v, CuVector<BaseFloat> *deriv_w,
    CuVector<BaseFloat> *deriv_S_and_b);

}  // namespace kaldi

#endif

// transform/mllt.h

// Copyright 2009-2011 Microsoft Corporation

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


#ifndef KALDI_TRANSFORM_MLLT_H_
#define KALDI_TRANSFORM_MLLT_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "transform/transform-common.h"
#include "transform/regression-tree.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"



namespace kaldi {


/** A class for estimating Maximum Likelihood Linear Transform, also known
    as global Semi-tied Covariance (STC), for GMMs.
    The resulting transform left-multiplies the feature vector.
*/
class MlltAccs {
 public:
  MlltAccs(): rand_prune_(0.0), beta_(0.0) { }

  /// Need rand_prune >= 0.
  /// The larger it is, the faster it will be.  Zero is exact.
  /// If a posterior p < rand_prune, will set p to
  /// rand_prune with probability (p/rand_prune), otherwise zero.
  /// E.g. 10 will give 10x speedup.
  MlltAccs(int32 dim, BaseFloat rand_prune = 0.25) { Init(dim, rand_prune); }

  /// initializes (destroys anything that was there before).
  void Init(int32 dim, BaseFloat rand_prune = 0.25);

  void Read(std::istream &is, bool binary, bool add = false);

  void Write(std::ostream &os, bool binary) const;

  int32 Dim() { return G_.size(); };  // returns model dimension.

  /// The Update function does the ML update; it requires that M has the
  /// right size.
  ///  @param [in, out] M  The output transform, will be of dimension Dim() x Dim().
  ///                   At input, should be the unit transform (the objective function
  ///                   improvement is measured relative to this value).
  ///  @param [out] objf_impr_out  The objective function improvement
  ///  @param [out] count_out  The data-count
  void Update(MatrixBase<BaseFloat> *M,
              BaseFloat *objf_impr_out,
              BaseFloat *count_out) const {
    Update(beta_, G_, M, objf_impr_out, count_out);
  }

  // A static version of the Update function, so it can
  // be called externally, given the right stats.
  static void Update(double beta,
                     const std::vector<SpMatrix<double> > &G,
                     MatrixBase<BaseFloat> *M,
                     BaseFloat *objf_impr_out,
                     BaseFloat *count_out);


  void AccumulateFromPosteriors(const DiagGmm &gmm,
                                const VectorBase<BaseFloat> &data,
                                const VectorBase<BaseFloat> &posteriors);

  // Returns GMM likelihood.
  BaseFloat AccumulateFromGmm(const DiagGmm &gmm,
                              const VectorBase<BaseFloat> &data,
                              BaseFloat weight);  // e.g. weight = 1.0

  BaseFloat AccumulateFromGmmPreselect(const DiagGmm &gmm,
                                       const std::vector<int32> &gselect,
                                       const VectorBase<BaseFloat> &data,
                                       BaseFloat weight);  // e.g. weight = 1.0

  
  // premultiplies the means of the model by M.  typically called
  // after update.
  // removed since we now do this using different code.
  // static void MultiplyGmmMeans(const Matrix<BaseFloat> &M,
  //  DiagGmm *gmm);

  /// rand_prune_ controls randomized pruning; the larger it is, the
  /// more pruning we do.  Typical value is 0.1.
  BaseFloat rand_prune_;
  double beta_;  // count.
  std::vector<SpMatrix<double> > G_;  // the G matrices (d matrices of size d x d)
};

} // namespace kaldi

#endif  // KALDI_TRANSFORM_MLLT_H_

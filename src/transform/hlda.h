// transform/hlda.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_TRANSFORM_HLDA_H_
#define KALDI_TRANSFORM_HLDA_H_

#include <vector>
#include <string>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/am-diag-gmm.h"
#include "transform/transform-common.h"
#include "transform/regression-tree.h"

namespace kaldi {


/// This class stores the compact form of the HLDA statistics, given a diagonal
/// GMM.  That it, it stores full-dimensional means for all the model's means,
/// and a set of global matrices for the full variance.  C.f. \ref transform_hlda
class HldaAccsDiagGmm {
 public:
  HldaAccsDiagGmm() { }

  /// Constructor; calls Init().
  HldaAccsDiagGmm(const AmDiagGmm &model,
                  int32 orig_feat_dim,
                  BaseFloat speedup = 1.0) {
    Init(model, orig_feat_dim, speedup);
  }

  int32 ModelDim() { return (S_.empty() ? 0 : S_.size()-1); }

  int32 FeatureDim() { return (S_.empty() ? 0 : S_[0].NumRows()); }

  /// Initializes the model.  Requires orig_feat_dim >= model.Dim()
  /// If speedup < 1.0, it will run a faster form of training in which it only
  /// accumulates the full stats from a subset of the data whose size
  /// is proportional to speedup (1.0 effectively uses all the data).
  void Init(const AmDiagGmm &model,
            int32 orig_feat_dim,
            BaseFloat speedup = 1.0);

  void Read(std::istream &is, bool binary, bool add = false);

  void Write(std::ostream &os, bool binary) const;

  /// The Update function does the ML update.  It outputs the appropriate transform
  /// and sets the model's means (in this approach, the model's variances are
  /// unchanged and you would have to run more passes of model re-estimation, and
  /// HLDA accumulation and update, if you want the model's variances to be
  /// trained.
  ///  @param model [in, out] The model, which must be the same model used to
  ///                     accumulate stats, or the update will be wrong.
  ///                     This function will set the model's means.
  ///  @param Mfull [in, out] Will be interpreted at input as the original full
  ///                     transform, of dimension orig-dim x orig-dim (e.g. from LDA),
  ///                     or previous iteration of HLDA.   Will be set at output to
  //                      the full transform of dimension orig-dim x orig-dim.
  ///  @param M [out] The output transform (only accepted rows), should be of dimension
  ///                  (feature-dim x orig-dim)
  ///  @param objf_impr_out [out] The objective function improvement
  ///  @param count_out [out] The data-count
  void Update(AmDiagGmm *model,
              MatrixBase<BaseFloat> *Mfull,
              MatrixBase<BaseFloat> *M,
              BaseFloat *objf_impr_out,
              BaseFloat *count_out) const;

  /// Accumulates stats (you have to first work out the posteriors yourself).
  void AccumulateFromPosteriors(int32 pdf_id,
                                const DiagGmm &gmm,
                                const VectorBase<BaseFloat> &data,
                                const VectorBase<BaseFloat> &posteriors);


 private:
  std::vector<SpMatrix<double> > S_;  // the S matrices: [model-dim+1] matrices of size (feat-dim) x (feat-dim)
  // are used to construct the G matrices.
  std::vector<Vector<double> > occs_;  // occupancies for the Gaussians. [num-pdfs][gauss]
  std::vector<Matrix<double> > mean_accs_;  // [num-pdfs][gauss][feat_dim+1]
  // occs_sub_ and mean_accs_sub_ are only used if speedup_ != 1.0.
  // occs_sub_ and mean_accs_sub_ are as occs_ and mean_accs, but accumulated
  // using just a subset of the data if we do the randomized speedup.
  BaseFloat speedup_;
  std::vector<Vector<double> > occs_sub_;
  std::vector<Matrix<double> > mean_accs_sub_;
  BaseFloat sample_gconst_;  // a sample gconst from the model, as a check
  // that the user does not switch the model between accu and update.
};



}  /// namespace kaldi

#endif  // KALDI_TRANSFORM_HLDA_H_

// transform/fmpe.h

// Copyright 2011-2012  Yanmin Qian  Johns Hopkins University (Author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_FMPE_H_
#define KALDI_TRANSFORM_FMPE_H_ 1

#include <vector>

#include "gmm/am-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

namespace kaldi {


struct FmpeOptions {
  // Probably the easiest place to start, to understand fMPE, is the
  // paper "Improvements to fMPE for discriminative training of features".
  // We are simplifying a few things here.  We are getting rid of the
  // "indirect differential"; we are adding a linear transform after the
  // high->low dimension projection whose function is to "un-whiten" the
  // transformed features (i.e. project from a nominally Gaussian-distributed
  // space into our actual feature space), in order to make it unnecessary to
  // take into account the per-dim variance during the update phase of fMPE;
  // and the update equations are rather simpler than described in
  // the paper; we take away some stuff, but add in the capability to
  // do l2 regularization during the update phase.
  
  std::string context_expansion; // This string describes the various contexts...
  // the easiest way to think of it is, we first generate the high-dimensional
  // features without context expansion, and we then append the left and right
  // frames, and also weighted averages of further-out frames, as specified by
  // this string.  Suppose there are 1024 Gaussians and the feature dimension is
  // 40.  In the simple way to describe it, supposing there are 9 contexts (the
  // central frame, the left and right frames, and 6 averages of more distant
  // frames), we generate the "offset features" of dimension (1024 * 41), then
  // add left and right temporal context to the high-dim features so the
  // dimension is (1024 * 41 * 9), and then project down to 40, so we train a
  // matrix of 40 x (1024 * 41 * 9).  As described in the paper, though, we
  // reorganize the computation for efficiency (it has to do with preserving
  // sparsity), and we train a matrix of dimension (40 * 9) x (1024 * 41).  The
  // (40 x 9) -> 40 transformation, which involves time as well, is dictated by
  // these contexts.

  // You probably won't want to mess with this "context_expansion" string.
  // The most important parameter to tune is the number of Gaussians in
  // the UBM.  Typically this will be in the range 300 to 1000.

  BaseFloat post_scale; // Scale on the posterior component of the high-dim
  // features (1 of these for every [feat-dim] of the offset features).
  // Typically 5.0-- this just gives a bit more emphasis to these posteriors
  // during training, like a faster learning rate.
  
  FmpeOptions(): context_expansion("0,1.0:-1,1.0:1,1.0:-2,0.5;-3,0.5:2,0.5;3,0.5:-4,0.5;-5,0.5:4,0.5;5,0.5:-6,0.333;-7,0.333;-8,0.333:6,0.333;7,0.333;8,0.333"),
                post_scale(5.0) { }

  void Register(OptionsItf *po) {
    po->Register("post-scale", &post_scale, "Scaling constant on posterior "
                 "element of offset features, to give it a faster learning "
                 "rate.");
    po->Register("context-expansion", &context_expansion, "Specifies the "
                 "temporal context-splicing of high-dimensional features.");
  }
  // We include write and read functions, since this
  // object is included as a member of the fMPE object.
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

struct FmpeUpdateOptions {
  BaseFloat learning_rate; // Learning rate constant.  Like inverse of E
  // in the papers.
  BaseFloat l2_weight; // Weight on l2 regularization term
  
  FmpeUpdateOptions(): learning_rate(0.1), l2_weight(100.0) { }

  void Register(OptionsItf *po) {
    po->Register("learning-rate", &learning_rate,
                 "Learning rate constant (like inverse of E in fMPE papers)");
    po->Register("l2-weight", &l2_weight,
                 "Weight on l2 regularization term in objective function.");
  }  
};

class Fmpe;

struct FmpeStats {
  FmpeStats() { };
  void Init(const Fmpe &fmpe);
  FmpeStats(const Fmpe &fmpe) { Init(fmpe); }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary, bool add = false);
  
  SubMatrix<BaseFloat> DerivPlus() const;
  SubMatrix<BaseFloat> DerivMinus() const;

  /// If we're using the indirect differential, accumulates certain quantities
  /// that will be used in the update phase to verify that the computation
  /// of the indirect differential was done correctly
  void AccumulateChecks(const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> &direct_deriv,
                        const MatrixBase<BaseFloat> &indirect_deriv);
  void DoChecks(); // Will check that stuff cancels.  Just prints
  // messages for now.
 private:
  Matrix<BaseFloat> deriv; // contains positive and negative parts of derivatives
  // separately as sub-parts of the matrix, to ensure memory locality.

  // checks() is an 8 x fmpe.FeatDim() matrix that stores:
  //  (0-1) summed-deriv from direct, +ve and -ve part.
  //  (2-3) summed-deriv from indirect, +ve and -ve part.
  //  (4-5) (summed-deriv from direct * features), +ve and -ve part.
  //  (6-7) (summed-deriv from indirect * features), +ve and -ve part.
  Matrix<double> checks; // contains quantities we use to check the
  // indirect and direct derivatives are canceling as they should.

};

class Fmpe {
 public:
  Fmpe() {}
  Fmpe(const DiagGmm &gmm, const FmpeOptions &config);

  int32 FeatDim() const { return gmm_.Dim(); }
  int32 NumGauss() const { return gmm_.NumGauss(); }
  int32 NumContexts() const { return static_cast<int32>(contexts_.size()); }

  // Note: this returns the number of rows and columns in projT_,
  // which is the transpose of the high->intermediate dimensional
  // projection matrix.  This is the dimension we want for the
  // stats.
  int32 ProjectionTNumRows() const { return (FeatDim()+1) * NumGauss(); }
  int32 ProjectionTNumCols() const { return FeatDim() * NumContexts(); }

  
  // Computes the fMPE feature offsets and outputs them.
  // You can add feat_in to this afterwards, if you want.
  // Requires the Gaussian-selection info, which would normally
  // be computed by a separate program-- this consists of
  // lists of the top-scoring Gaussians for these features.
  void ComputeFeatures(const MatrixBase<BaseFloat> &feat_in,
                       const std::vector<std::vector<int32> > &gselect,
                       Matrix<BaseFloat> *feat_out) const;

  // For training-- compute the derivative w.r.t the projection matrix
  // (we keep the positive and negative parts separately to help
  // set the learning rates).
  void AccStats(const MatrixBase<BaseFloat> &feat_in,
                const std::vector<std::vector<int32> > &gselect,
                const MatrixBase<BaseFloat> &direct_feat_deriv,
                const MatrixBase<BaseFloat> *indirect_feat_deriv, // may be NULL
                FmpeStats *stats) const;
  
  // Note: the form on disk starts with the GMM; that way,
  // the gselect program can treat the fMPE object as if it
  // is a GMM.
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Returns total objf improvement, based on linear assumption.
  BaseFloat Update(const FmpeUpdateOptions &config,
                   const FmpeStats &stats);
  
 private:
  void SetContexts(std::string context_str);
  void ComputeC(); // Computes the Cholesky factor C, from the GMM.
  void ComputeStddevs();

  // Constructs the high-dim features and applies the main projection matrix proj_.
  void ApplyProjection(const MatrixBase<BaseFloat> &feat_in,
                       const std::vector<std::vector<int32> > &gselect,
                       MatrixBase<BaseFloat> *intermed_feat) const;

  // The same in reverse, for computing derivatives.
  void ApplyProjectionReverse(const MatrixBase<BaseFloat> &feat_in,
                              const std::vector<std::vector<int32> > &gselect,
                              const MatrixBase<BaseFloat> &intermed_feat_deriv,
                              MatrixBase<BaseFloat> *proj_deriv_plus,
                              MatrixBase<BaseFloat> *proj_deriv_minus) const;

  // Applies the temporal context splicing from the intermediate
  // features-- adds the result to feat_out which at this point
  // will typically be zero.
  void ApplyContext(const MatrixBase<BaseFloat> &intermed_feat,
                    MatrixBase<BaseFloat> *feat_out) const;

  // This is as ApplyContext but for back-propagating the derivative.
  // Result is added to intermediate_feat_deriv which at this point will
  // typically be zero.
  void ApplyContextReverse(const MatrixBase<BaseFloat> &feat_deriv,
                           MatrixBase<BaseFloat> *intermed_feat_deriv) const;

  // Multiplies the feature offsets by the Cholesky matrix C.
  void ApplyC(MatrixBase<BaseFloat> *feat_out, bool reverse = false) const;

  // For computing derivatives-- multiply the derivatives by C^T,
  // which is the "reverse" of the forward pass of multiplying
  // by C (this is how derivatives behave...)
  void ApplyCReverse(MatrixBase<BaseFloat> *deriv) const { ApplyC(deriv, true); }

  
  
  DiagGmm gmm_; // The GMM used to get posteriors.
  FmpeOptions config_;
  Matrix<BaseFloat> stddevs_; // The standard deviations of the
  // variances of the GMM -- computed to avoid taking a square root
  // in the fMPE computation.   Derived variable-- not stored on
  // disk.
  Matrix<BaseFloat> projT_; // The transpose of the projection matrix;
  // this is of dimension
  // (NumGauss() * (FeatDim()+1)) * (FeatDim() * NumContexts()).
  
  TpMatrix<BaseFloat> C_; // Cholesky factor of the variance Sigma of
  // features around their mean (as estimated from GMM)... applied
  // to fMPE offset just before we add it to the features.  This allows
  // us to simplify the fMPE update and not have to worry about
  // the features having non-unit variance, and what effect this should
  // have on the learning rate..
  
  // The following variable dictates how we use temporal context.
  // e.g. contexts = { { (0, 1.0) }, { (-1, 1.0) }, { (1, 1.0) },
  //                   { (-2, 0.5 ), (-3, 0.5) }, ...  }
  std::vector<std::vector<std::pair<int32, BaseFloat> > > contexts_;
  
};

/// Computes derivatives of the likelihood of these states (weighted),
/// w.r.t. the feature values.  Used in fMPE training.  Note, the
/// weights "posterior" may be positive or negative-- for MMI, MPE,
/// etc., they will typically be of both signs.  Will resize "deriv".
/// Returns the sum of (GMM likelihood * weight), which may be used
/// as an approximation to the objective function.
/// Last two parameters are optional.  See GetStatsDerivative() for
/// or fMPE paper (ICASSP, 2005) more info on indirect derivative.
/// Caution: if you supply the last two parameters, this function only
/// works in the MMI case as it assumes the stats with positive weight
/// are numerator == ml stats-- this is only the same thing in the MMI
/// case, not fMPE.
BaseFloat ComputeAmGmmFeatureDeriv(const AmDiagGmm &am_gmm,
                                   const TransitionModel &trans_model,
                                   const Posterior &posterior,
                                   const MatrixBase<BaseFloat> &features,
                                   Matrix<BaseFloat> *direct_deriv,
                                   const AccumAmDiagGmm *model_diff = NULL,
                                   Matrix<BaseFloat> *indirect_deriv = NULL);



}  // End namespace kaldi


#endif

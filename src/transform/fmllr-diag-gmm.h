// transform/fmllr-diag-gmm.h

// Copyright 2009-2011 Microsoft Corporation  Arnab Ghoshal

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


#ifndef KALDI_TRANSFORM_FMLLR_DIAG_GMM_H_
#define KALDI_TRANSFORM_FMLLR_DIAG_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "transform/transform-common.h"
#include "transform/regression-tree.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"
#include "transform/fmllr-diag-gmm.h"


namespace kaldi {

/* This header contains routines for performing global CMLLR,
   without a regression tree (however, you can down-weight silence
   in training using the program weight-silence-post on the
   state-level posteriors).  For regression-tree CMLLR, see
   fmllr-diag-gmm.h
*/

struct FmllrOptions {
  std::string update_type;  ///< "full", "diag", "offset", "none"
  BaseFloat min_count;
  int32 num_iters;
  FmllrOptions(): update_type("full"), min_count(500.0), num_iters(40) { }
  void Register(ParseOptions *po) {
    po->Register("fmllr-update-type", &update_type,
                 "Update type for fMLLR (\"full\"|\"diag\"|\"offset\"|\"none\")");
    po->Register("fmllr-min-count", &min_count,
                 "Minimum count required to update fMLLR");
    po->Register("fmllr-num-iters", &num_iters,
                 "Number of iterations in fMLLR update phase.");
  }
};


/// This does not work with multiple feature transforms.

class FmllrDiagGmmAccs: public AffineXformStats {
 public:
  FmllrDiagGmmAccs() { }
  FmllrDiagGmmAccs(const FmllrDiagGmmAccs &other):
      AffineXformStats(other) { }
  explicit FmllrDiagGmmAccs(size_t dim) { Init(dim); }
  void Init(size_t dim) { AffineXformStats::Init(dim, dim); }

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  BaseFloat AccumulateForGmm(const DiagGmm &gmm,
                             const VectorBase<BaseFloat>& data,
                             BaseFloat weight);

  /// Accumulate stats for a single Gaussian component in the model.
  void AccumulateFromPosteriors(const DiagGmm &gmm,
                                const VectorBase<BaseFloat>& data,
                                const VectorBase<BaseFloat> &posteriors);

  void Update(const FmllrOptions &opts,
              MatrixBase<BaseFloat> *fmllr_mat,
              BaseFloat *objf_impr,
              BaseFloat *count) const;

  // Note: we allow copy and assignment for this class.
};


// Initializes the FMLLR matrix to its default values.
inline void InitFmllr(int32 dim,
                            Matrix<BaseFloat> *out_fmllr) {
  out_fmllr->Resize(dim, dim+1);
  out_fmllr->SetUnit();  // sets diagonal elements to one.
}

// ComputeFmllr optimizes the FMLLR matrix, controlled by the options.
// It starts the optimization from the current value of the matrix (e.g. use
// InitFmllr to get this).
// Returns auxf improvement.
BaseFloat ComputeFmllrDiagGmm(const FmllrDiagGmmAccs& accs,
                                    const FmllrOptions &opts,
                                    Matrix<BaseFloat> *out_fmllr,
                                    BaseFloat *logdet);  // add this to likelihoods

inline BaseFloat ComputeFmllrLogDet(const Matrix<BaseFloat> &fmllr_mat) {
  KALDI_ASSERT(fmllr_mat.NumRows() != 0 && fmllr_mat.NumCols() == fmllr_mat.NumRows()+1);
  SubMatrix<BaseFloat> tmp(fmllr_mat,
                           0, fmllr_mat.NumRows(),
                           0, fmllr_mat.NumRows());
  return tmp.LogDet();
}


/// Updates the FMLLR matrix using Mark Gales' row-by-row update.
/// Uses full fMLLR matrix (no structure).  Returns the
/// objective function improvement, not normalized by number of frames.
BaseFloat ComputeFmllrMatrixDiagGmmFull(const MatrixBase<BaseFloat> &in_xform,
                                        const AffineXformStats& stats,
                                        int32 num_iters,
                                        MatrixBase<BaseFloat> *out_xform);

/// This does diagonal fMLLR (i.e. only estimate an offset and scale per
/// dimension).  The format of the output is the same as for the full case.  Of
/// course, these statistics are unnecessarily large for this case.  Returns the
/// objective function improvement, not normalized by number of frames.
BaseFloat ComputeFmllrMatrixDiagGmmDiagonal(const MatrixBase<BaseFloat> &in_xform,
                                            const AffineXformStats& stats,
                                            MatrixBase<BaseFloat> *out_xform);

/// This does offset-only fMLLR, i.e. it only estimates an offset.
BaseFloat ComputeFmllrMatrixDiagGmmOffset(const MatrixBase<BaseFloat> &in_xform,
                                          const AffineXformStats& stats,
                                          MatrixBase<BaseFloat> *out_xform);


/// This function internally calls ComputeFmllrMatrixDiagGmm{Full, Diagonal, Offset},
/// depending on "fmllr_type".
BaseFloat ComputeFmllrMatrixDiagGmm(const MatrixBase<BaseFloat> &in_xform,
                                    const AffineXformStats& stats,
                                    std::string fmllr_type,  // "none", "offset", "diag", "full"
                                    int32 num_iters,
                                    MatrixBase<BaseFloat> *out_xform);



/// Updates the FMLLR matrix using gradient descent.  Returns the objective
/// function improvement, not normalized by number of frames.
BaseFloat ComputeFmllrMatrixDiagGmmGradient(const MatrixBase<BaseFloat> &in_xform,
                                            const AffineXformStats& stats,
                                            int32 num_iters,
                                            MatrixBase<BaseFloat> *out_xform);

/// Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform
/// and the stats.
float FmllrAuxFuncDiagGmm(const MatrixBase<float> &xform,
                              const AffineXformStats& stats);
double FmllrAuxFuncDiagGmm(const MatrixBase<double> &xform,
                           const AffineXformStats& stats);



/// Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform
/// and the stats.
BaseFloat FmllrAuxfGradient(const MatrixBase<BaseFloat> &xform,
                            const AffineXformStats& stats,
                            MatrixBase<BaseFloat> *grad_out);


/// This is only really useful when dealing with a global fMLLR transform
/// (see fmllr-diag-gmm-global.h)
void ApplyFmllrTransform(const MatrixBase<BaseFloat> &xform,
                         VectorBase<BaseFloat> *feats);


/// This function applies a feature-level transform to stats (useful for
/// certain techniques based on fMLLR).  Assumes the stats are of the
/// standard diagonal sort.
/// The transform "xform" may be either dim x dim (linear),
/// dim x dim+1 (affine), or dim+1 x dim+1 (affine with the
/// last row equal to 0 0 0 .. 0 1).
void ApplyFeatureTransformToStats(const MatrixBase<BaseFloat> &xform,
                                  AffineXformStats *stats);

/// ApplyModelTransformToStats takes a transform "xform", which must be diagonal
/// (i.e. of the form T = [ D; b ] where D is diagonal), and applies it to the
/// stats as if we had made it a model-space transform (note that the transform
/// applied to the model means is the inverse transform of T).  Thus, if we are
/// estimating a transform T U, and we get stats valid for estimating T U and we
/// estimate T, we can then call this function (treating T as a model-space
/// transform) and will get stats valid for estimating U.  This only works if T is
/// diagonal, because otherwise the standard stats format is not valid.  xform must
/// be of dimension d x d+1
void ApplyModelTransformToStats(const MatrixBase<BaseFloat> &xform,
                                AffineXformStats *stats);


} // namespace kaldi

#endif  // KALDI_TRANSFORM_FMLLR_DIAG_GMM_H_

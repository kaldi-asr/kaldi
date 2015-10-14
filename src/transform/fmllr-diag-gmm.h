// transform/fmllr-diag-gmm.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2013  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_FMLLR_DIAG_GMM_H_
#define KALDI_TRANSFORM_FMLLR_DIAG_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "transform/transform-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"

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
  void Register(OptionsItf *opts) {
    opts->Register("fmllr-update-type", &update_type,
                   "Update type for fMLLR (\"full\"|\"diag\"|\"offset\"|\"none\")");
    opts->Register("fmllr-min-count", &min_count,
                   "Minimum count required to update fMLLR");
    opts->Register("fmllr-num-iters", &num_iters,
                   "Number of iterations in fMLLR update phase.");
  }
};


/// This does not work with multiple feature transforms.

class FmllrDiagGmmAccs: public AffineXformStats {
 public:
  // If supplied, the "opts" will only be used to limit the
  // stats that are accumulated, to the parts we'll need in the
  // update.
  FmllrDiagGmmAccs(const FmllrOptions &opts = FmllrOptions()):
      opts_(opts) { }
  explicit FmllrDiagGmmAccs(const FmllrDiagGmmAccs &other):
      AffineXformStats(other), single_frame_stats_(other.single_frame_stats_),
      opts_(other.opts_) {}
  explicit FmllrDiagGmmAccs(int32 dim, const FmllrOptions &opts = FmllrOptions()):
      opts_(opts) { Init(dim); }
  
  // The following initializer gives us an efficient way to
  // compute these stats from full-cov Gaussian statistics
  // (accumulated from a *diagonal* model (e.g. use
  // AccumFullGmm::AccumulateFromPosteriors or
  // AccumulateFromDiag).
  FmllrDiagGmmAccs(const DiagGmm &gmm, const AccumFullGmm &fgmm_accs);
  
  void Init(size_t dim) {
    AffineXformStats::Init(dim, dim); single_frame_stats_.Init(dim);
  }
  void Read(std::istream &in, bool binary, bool add) {
      AffineXformStats::Read(in, binary, add);
      single_frame_stats_.Init(Dim());
  }
  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  BaseFloat AccumulateForGmm(const DiagGmm &gmm,
                             const VectorBase<BaseFloat> &data,
                             BaseFloat weight);

  /// This is like AccumulateForGmm but when you have gselect
  /// (Gaussian selection) information
  BaseFloat AccumulateForGmmPreselect(const DiagGmm &gmm,
                                      const std::vector<int32> &gselect,
                                      const VectorBase<BaseFloat> &data,
                                      BaseFloat weight);
  
  /// Accumulate stats for a GMM, given supplied posteriors.
  void AccumulateFromPosteriors(const DiagGmm &gmm,
                                const VectorBase<BaseFloat> &data,
                                const VectorBase<BaseFloat> &posteriors);

  /// Accumulate stats for a GMM, given supplied posteriors.  The "posteriors"
  /// vector should be have the same size as "gselect".
  void AccumulateFromPosteriorsPreselect(
      const DiagGmm &gmm,
      const std::vector<int32> &gselect,
      const VectorBase<BaseFloat> &data,
      const VectorBase<BaseFloat> &posteriors);

  
  /// Update
  void Update(const FmllrOptions &opts,
              MatrixBase<BaseFloat> *fmllr_mat,
              BaseFloat *objf_impr,
              BaseFloat *count);

  // Note: we allow copy and assignment for this class.

  // Note: you can use the inherited AffineXformStats::Read 
  //       and AffineXformStats::Write methods for writing/reading
  //       of the object. It is not necessary to store the other 
  //       private variables of this class
       
 private:
  // The things below, added in 2013, relate to an optimization that lets us
  // speed up accumulation if there are multiple active pdfs per frame
  // (e.g. when accumulating from lattices), or if we don't anticipate
  // doing a "full" update.
  
  struct SingleFrameStats {
    Vector<BaseFloat> x; // dim-dimensional features.
    Vector<BaseFloat> a; // linear term in per-frame auxf; dim is model-dim.
    Vector<BaseFloat> b; // quadratic term in per-frame auxf; dim is model-dim.
    double count;
    SingleFrameStats(int32 dim = 0) { Init(dim); }
    SingleFrameStats(const SingleFrameStats &s): x(s.x), a(s.a), b(s.b),
                                                 count(s.count) {}
    void Init(int32 dim);
  };  

  void CommitSingleFrameStats();

  void InitSingleFrameStats(const VectorBase<BaseFloat> &data);
  
  bool DataHasChanged(const VectorBase<BaseFloat> &data) const; // compares it to the
  // data in single_frame_stats_, returns true if it's different.

  SingleFrameStats single_frame_stats_;
  
  // We only use the opts_ variable for its "update_type" data member,
  // which limits what parts of the G matrix we accumulate.
  FmllrOptions opts_;
  
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
BaseFloat ComputeFmllrDiagGmm(const FmllrDiagGmmAccs &accs,
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
                                        const AffineXformStats &stats,
                                        int32 num_iters,
                                        MatrixBase<BaseFloat> *out_xform);

/// This does diagonal fMLLR (i.e. only estimate an offset and scale per
/// dimension).  The format of the output is the same as for the full case.  Of
/// course, these statistics are unnecessarily large for this case.  Returns the
/// objective function improvement, not normalized by number of frames.
BaseFloat ComputeFmllrMatrixDiagGmmDiagonal(const MatrixBase<BaseFloat> &in_xform,
                                            const AffineXformStats &stats,
                                            MatrixBase<BaseFloat> *out_xform);
// Simpler implementation I am testing.
BaseFloat ComputeFmllrMatrixDiagGmmDiagonal2(const MatrixBase<BaseFloat> &in_xform,
                                             const AffineXformStats &stats,
                                             MatrixBase<BaseFloat> *out_xform);

/// This does offset-only fMLLR, i.e. it only estimates an offset.
BaseFloat ComputeFmllrMatrixDiagGmmOffset(const MatrixBase<BaseFloat> &in_xform,
                                          const AffineXformStats &stats,
                                          MatrixBase<BaseFloat> *out_xform);


/// This function internally calls ComputeFmllrMatrixDiagGmm{Full, Diagonal, Offset},
/// depending on "fmllr_type".
BaseFloat ComputeFmllrMatrixDiagGmm(const MatrixBase<BaseFloat> &in_xform,
                                    const AffineXformStats &stats,
                                    std::string fmllr_type,  // "none", "offset", "diag", "full"
                                    int32 num_iters,
                                    MatrixBase<BaseFloat> *out_xform);

/// Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform
/// and the stats.
float FmllrAuxFuncDiagGmm(const MatrixBase<float> &xform,
                          const AffineXformStats &stats);
double FmllrAuxFuncDiagGmm(const MatrixBase<double> &xform,
                           const AffineXformStats &stats);



/// Returns the (diagonal-GMM) FMLLR auxiliary function value given the transform
/// and the stats.
BaseFloat FmllrAuxfGradient(const MatrixBase<BaseFloat> &xform,
                            const AffineXformStats &stats,
                            MatrixBase<BaseFloat> *grad_out);


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


/// This function does one row of the inner-loop fMLLR transform update.
/// We export it because it's needed in the RawFmllr code.
/// Here, if inv_G is the inverse of the G matrix indexed by this row,
/// and k is the corresponding row of the K matrix.
void FmllrInnerUpdate(SpMatrix<double> &inv_G,
                      VectorBase<double> &k,
                      double beta,
                      int32 row,
                      MatrixBase<double> *transform);

                      
                      


} // namespace kaldi

#endif  // KALDI_TRANSFORM_FMLLR_DIAG_GMM_H_

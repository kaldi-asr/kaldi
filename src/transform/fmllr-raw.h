// transform/fmllr-raw.h

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_FMLLR_RAW_H_
#define KALDI_TRANSFORM_FMLLR_RAW_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "transform/transform-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"

namespace kaldi {


/*
  This header contains classes and functions related to computing Constrained
  MLLR (equivalently, fMLLR) on the raw MFCCs or similar, when they have been
  spliced and projected with something like LDA+MLLT, but where our model is
  built on top of the spliced and projected features.  We back-project the
  model estimation back to the original MFCCs so our transform optimizes the
  data likelihood given our model in the projected space.  We have to include
  the rejected dimensions in this likelihood, too.  The objective function
  includes N times the log-determinant of the square part of the transform,
  where N is the number of times we spliced consecutive features (e.g. N = 9,
  if we spliced +- 4 frames of context).

  For concreteness (but without losing generality), assuming we spliced
  13-dimensional MFCCs across 9 frames to get 117-dimensional features.
   
  Each of the 117-dim features is a linear function of the 13(13+1) transform
  parameters.  We have a particular vectorization of these parameters, from
  which (with the transform) we work out the full quadratic auxiliary function
  w.r.t. the parameters.

  This gives us a generic quadratic scalar function of the 13(13+1) parameters.
  How to get this quadratic w.r.t. one row?  Always keep updated the current
  derivative w.r.t. one row.  The quadratic w.r.t. that row can be read off.
  The log-determinant is easy to work out from the cofactor.

  So the full stats will be a (13(13+1)) by (13(13+1)) SpMatrix, plus
  a bias term.

  The update will iterate row by row, and work out the quadratic function
  of the row.
*/


struct FmllrRawOptions {
  BaseFloat min_count;
  int32 num_iters;
  FmllrRawOptions(): min_count(100.0), num_iters(20) { }
  void Register(OptionsItf *po) {
    po->Register("fmllr-min-count", &min_count,
                 "Minimum count required to update fMLLR");
    po->Register("fmllr-num-iters", &num_iters,
                 "Number of iterations in fMLLR update phase.");
  }
};

class FmllrRawAccs {
 public:
  FmllrRawAccs() { }

  /// Dimension of raw MFCC (etc.) features
  int32 RawDim() const { return raw_dim_; }
  /// Full feature dimension after splicing.
  int32 FullDim() const { return full_transform_.NumRows(); }
  /// Number of frames that are spliced together each time.
  int32 SpliceWidth() const { return FullDim() / RawDim(); }
  /// Dimension of the model.
  int32 ModelDim() const { return model_dim_; }
  
  // Initializer takes the raw dimension of the features (e.g. 13 for typicaly
  // MFCC features, and the full transform (e.g. an LDA+MLLT transform).  This
  // full transform is the transform extended with the "rejected rows" that
  // we would normally discard; we need them for this type of estimation.
  FmllrRawAccs(int32 raw_dim,
               int32 model_dim,
               const Matrix<BaseFloat> &full_transform);

  
  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// Here, "data" will typically be of larger dimension than the model.
  /// Note: "data" is the original, spliced features-- before LDA+MLLT.
  /// Returns log-like for this data given this GMM, including rejected
  /// dimensions (not multiplied by weight).
  BaseFloat AccumulateForGmm(const DiagGmm &gmm,
                             const VectorBase<BaseFloat> &data,
                             BaseFloat weight);
  
  /// Accumulate stats for a GMM, given supplied posteriors.  Note: "data" is
  /// the original, spliced features-- before LDA+MLLT. 
  void AccumulateFromPosteriors(const DiagGmm &gmm,
                                const VectorBase<BaseFloat> &data,
                                const VectorBase<BaseFloat> &posteriors);

  /// Update "raw_fmllr_mat"; it should have the correct dimension and
  /// reasonable values at entry (see the function InitFmllr in fmllr-diag-gmm.h
  /// for how to initialize it.)
  /// The only reason this function is not const is because we may have
  /// to call CommitSingleFrameStats().
  void Update(const FmllrRawOptions &opts,
              MatrixBase<BaseFloat> *raw_fmllr_mat,
              BaseFloat *objf_impr,
              BaseFloat *count);

  void SetZero();
 private:
  struct SingleFrameStats {
    Vector<BaseFloat> s; // [FullDim() + 1]-dimensional spliced data, plus 1.0
    Vector<BaseFloat> transformed_data; // [FullDim()] Data times full transform, with offset.
    double count;
    Vector<double> a; // linear term in per-frame auxf; dim is model-dim.
    Vector<double> b; // quadratic term in per-frame auxf; dim is model-dim.
  };
  
  void CommitSingleFrameStats();

  void InitSingleFrameStats(const VectorBase<BaseFloat> &data);
  
  bool DataHasChanged(const VectorBase<BaseFloat> &data) const; // compares it to the
  // data in single_frame_stats_, returns true if it's different.

  
  /// Compute the auxiliary function for this matrix.
  double GetAuxf(const Vector<double> &simple_linear_stats,
                 const SpMatrix<double> &simple_quadratic_stats,
                 const Matrix<double> &fmllr_mat) const;

  /// Converts from the Q and S stats to a simple objective function
  /// of the form l . simple_linear_stats -0.5 l^t simple_quadratic_stats l,
  /// plus the determinant term, where l is the linearized transform.
  void ConvertToSimpleStats(
      Vector<double> *simple_linear_stats,
      SpMatrix<double> *simple_quadratic_stats) const;

  /// Computes the M_i matrices used in the update, see the extended comment in
  /// fmllr-raw.cc for explanation.
  void ComputeM(
      std::vector<Matrix<double> > *M) const;
  
  /// Transform stats into a convenient format for the update.
  /// linear_stats is of dim RawDim() by RawDim() + 1, it's the linear term.
  /// diag_stats (of dimension RawDim(), each element of dimension RawDim() + 1
  /// is the quadratic terms w.r.t. the diagonals.  off_diag_stats contains the
  /// cross-terms between different rows; it is indexed [i][j], with
  /// 0 <= i < RawDim(), and j < i, and each element is of dimension RawDim() + 1
  /// by RawDim() + 1.  The [i][j]'th element is interpreted as follows:
  /// the inner product with the [i'th row] [element [i][j]] [j'th row] is the
  /// term in the objective function.
  /// This function resizes its output.
  void ConvertToPerRowStats(
      const Vector<double> &simple_linear_stats,
      const SpMatrix<double> &simple_quadratic_stats_sp,
      Matrix<double> *linear_stats,
      std::vector<SpMatrix<double> > *diag_stats,
      std::vector<std::vector<Matrix<double> > > *off_diag_stats) const;
  
  int32 raw_dim_; // Raw MFCC dimension.
  int32 model_dim_; // Model dimension

  Matrix<BaseFloat> full_transform_; // Does not include any offset term
  // (last column).
  Vector<BaseFloat> transform_offset_; // The offset term (or zero).
  

  SingleFrameStats single_frame_stats_;
  
  double count_; // The data-count.  Note: in accounting for the determinant, we will
                 // have to multiply this by the number of times the data is spliced
                 // together on each frame.

  SpMatrix<double> temp_; // [full_dim + 1][full_dim + 1], outer product of s.
  Matrix<double> Q_; // linear stats, indexed [model_dim + 1][full_dim + 1]
  Matrix<double> S_; // quadratic stats, indexed
                     // [model_dim + 1][((full_dim+1)*(full_dim+2))/2]
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(FmllrRawAccs);
};



} // namespace kaldi

#endif  // KALDI_TRANSFORM_FMLLR_RAW_H_

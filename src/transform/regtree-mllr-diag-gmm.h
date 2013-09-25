// transform/regtree-mllr-diag-gmm.h

// Copyright 2009-2011  Saarland University;  Jan Silovsky

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

#ifndef KALDI_TRANSFORM_REGTREE_MLLR_DIAG_GMM_H_
#define KALDI_TRANSFORM_REGTREE_MLLR_DIAG_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "transform/transform-common.h"
#include "transform/regression-tree.h"
#include "util/common-utils.h"

namespace kaldi {


///  Configuration variables for FMLLR transforms
struct RegtreeMllrOptions {
  BaseFloat min_count;  ///< Minimum occupancy for computing a transform

  /// If 'true', find transforms to generate using regression tree.
  /// If 'false', generate transforms for each baseclass.
  bool use_regtree;

  RegtreeMllrOptions(): min_count(1000.0), use_regtree(true) { }

  void Register(OptionsItf *po) {
    po->Register("mllr-min-count", &min_count,
                 "Minimum count to estimate an MLLR transform.");
    po->Register("mllr-use-regtree", &use_regtree,
                 "Use a regression-class tree for MLLR.");
  }
};

/// An MLLR mean transformation is an affine transformation of Gaussian means.
class RegtreeMllrDiagGmm {
 public:
  RegtreeMllrDiagGmm() {}

  /// Allocates memory for transform matrix & bias vector
  void Init(int32 num_xforms, int32 dim);

  /// Initialize transform matrix to identity and bias vector to zero
  void SetUnit();

  /// Apply the transform(s) to all the Gaussian means in the model
  void TransformModel(const RegressionTree &regtree, AmDiagGmm *am);

  /// Get all the transformed means for a given pdf.
  void GetTransformedMeans(const RegressionTree &regtree, const AmDiagGmm &am,
                           int32 pdf_index, MatrixBase<BaseFloat> *out) const;

  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary);

  /// Mutators
  void SetParameters(const MatrixBase<BaseFloat> &mat, int32 regclass);
  void set_bclass2xforms(const std::vector<int32> &in) { bclass2xforms_ = in; }

  /// Accessors
  const std::vector< Matrix<BaseFloat> > xform_matrices() const {
    return xform_matrices_;
  }

 private:
  /// Transform matrices: size() = num_xforms_
  std::vector< Matrix<BaseFloat> > xform_matrices_;
  int32 num_xforms_;  ///< Number of transforms == xform_matrices_.size()
  /// For each baseclass index of which transform to use; -1 => no xform
  std::vector<int32> bclass2xforms_;
  int32 dim_;  ///< Dimension of feature vectors

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(RegtreeMllrDiagGmm);
};

inline void RegtreeMllrDiagGmm::SetParameters(const MatrixBase<BaseFloat> &mat,
                                              int32 regclass) {
  xform_matrices_[regclass].CopyFromMat(mat, kNoTrans);
}

/** Class for computing the maximum-likelihood estimates of the parameters of
 *  an acoustic model that uses diagonal Gaussian mixture models as emission
 *  densities.
 */
class RegtreeMllrDiagGmmAccs {
 public:
  RegtreeMllrDiagGmmAccs() {}
  ~RegtreeMllrDiagGmmAccs() { DeletePointers(&baseclass_stats_); }

  void Init(int32 num_bclass, int32 dim);
  void SetZero();

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work with multiple feature transforms.
  BaseFloat AccumulateForGmm(const RegressionTree &regtree,
                             const AmDiagGmm &am,
                             const VectorBase<BaseFloat> &data,
                             int32 pdf_index, BaseFloat weight);

  /// Accumulate stats for a single Gaussian component in the model.
  void AccumulateForGaussian(const RegressionTree &regtree,
                             const AmDiagGmm &am,
                             const VectorBase<BaseFloat> &data,
                             int32 pdf_index, int32 gauss_index,
                             BaseFloat weight);

  void Update(const RegressionTree &regtree, const RegtreeMllrOptions &opts,
              RegtreeMllrDiagGmm *out_mllr, BaseFloat *auxf_impr,
              BaseFloat *t) const;

  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary, bool add);

  /// Accessors
  int32 Dim() const { return dim_; }
  int32 NumBaseClasses() const { return num_baseclasses_; }
  const std::vector<AffineXformStats*> &baseclass_stats() const {
    return baseclass_stats_;
  }

 private:
  /// Per-baseclass stats; used for accumulation
  std::vector<AffineXformStats*> baseclass_stats_;
  int32 num_baseclasses_;    ///< Number of baseclasses
  int32 dim_;    ///< Dimension of feature vectors

  /// Returns the MLLR objective function for a given transform and baseclass.
  BaseFloat MllrObjFunction(const Matrix<BaseFloat> &xform,
                            int32 bclass_id) const;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(RegtreeMllrDiagGmmAccs);
};

typedef TableWriter< KaldiObjectHolder<RegtreeMllrDiagGmm> >
            RegtreeMllrDiagGmmWriter;
typedef RandomAccessTableReader< KaldiObjectHolder<RegtreeMllrDiagGmm> >
            RandomAccessRegtreeMllrDiagGmmReader;
typedef RandomAccessTableReaderMapped< KaldiObjectHolder<RegtreeMllrDiagGmm> >
            RandomAccessRegtreeMllrDiagGmmReaderMapped;
typedef SequentialTableReader< KaldiObjectHolder<RegtreeMllrDiagGmm> >
            RegtreeMllrDiagGmmSeqReader;

}  // namespace kaldi

#endif  // KALDI_TRANSFORM_REGTREE_MLLR_DIAG_GMM_H_

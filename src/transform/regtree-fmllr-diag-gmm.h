// transform/regtree-fmllr-diag-gmm.h

// Copyright 2009-2011  Saarland University;  Georg Stemmer;
//                      Microsoft Corporation

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


#ifndef KALDI_TRANSFORM_REGTREE_FMLLR_DIAG_GMM_H_
#define KALDI_TRANSFORM_REGTREE_FMLLR_DIAG_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "transform/transform-common.h"
#include "transform/regression-tree.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"

namespace kaldi {


///  Configuration variables for FMLLR transforms
struct RegtreeFmllrOptions {
  std::string update_type;  ///< "full", "diag", "offset", "none"
  BaseFloat min_count;  ///< Minimum occupancy for computing a transform
  int32 num_iters;      ///< Number of iterations (if using an iterative update)
  bool use_regtree;     ///< If 'true', find transforms to generate using regression tree.
                        ///< If 'false', generate transforms for each baseclass.

  RegtreeFmllrOptions(): update_type("full"), min_count(1000.0),
                         num_iters(10), use_regtree(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("fmllr-update-type", &update_type,
                   "Update type for fMLLR (\"full\"|\"diag\"|\"offset\"|\"none\")");
    opts->Register("fmllr-min-count", &min_count,
                   "Minimum count to estimate an fMLLR transform.");
    opts->Register("fmllr-num-iters", &num_iters,
                   "Number of fMLLR iterations (if using an iterative update).");
    opts->Register("fmllr-use-regtree", &use_regtree,
                   "Use a regression-class tree for fMLLR.");
  }
};


/** An FMLLR (feature-space MLLR) transformation, also called CMLLR
 *  (constrained MLLR) is an affine transformation of the feature vectors.
 *  This class supports multiple transforms, and a regression tree.
 *  For a single, feature-level transformation see fmllr-diag-gmm-global.h
 *  Note: the "regression classes" are the classes after tree-clustering,
 *  which are smaller in number than the "base classes"  (these correspond
 *  to the leaves of the tree).
 */
class RegtreeFmllrDiagGmm {
 public:
  RegtreeFmllrDiagGmm() : dim_(-1), num_xforms_(-1), valid_logdet_(false) {}
  explicit RegtreeFmllrDiagGmm(const RegtreeFmllrDiagGmm &other)
      : dim_(other.dim_), num_xforms_(other.num_xforms_),
        xform_matrices_(other.xform_matrices_), logdet_(other.logdet_),
        valid_logdet_(other.valid_logdet_),
        bclass2xforms_(other.bclass2xforms_) {}
  ~RegtreeFmllrDiagGmm() {}
  /// Allocates memory for transform matrix & bias vector
  void Init(size_t num_xforms, size_t dim);
  void Validate();  ///< Checks whether the various parameters are consistent
  /// Sets transform matrix to identity and bias vector to zero
  void SetUnit();
  /// Computes the log-determinant of the Jacobians for each transform
  void ComputeLogDets();
  /// Get the transformed features for each of the transforms.
  void TransformFeature(const VectorBase<BaseFloat> &in,
                        std::vector< Vector<BaseFloat> > *out) const;
  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary);

  /// Accessors
  int32 Dim() const { return dim_; }
  int32 NumBaseClasses() const { return bclass2xforms_.size(); }
  int32 NumRegClasses() const { return num_xforms_; }
  void GetXformMatrix(int32 xform_index, Matrix<BaseFloat> *out) const;
  void GetLogDets(VectorBase<BaseFloat> *out) const;
  int32 Base2RegClass(int32 bclass) const { return bclass2xforms_[bclass]; }

  /// Mutators
  void SetParameters(const MatrixBase<BaseFloat> &mat, size_t regclass);
  void set_bclass2xforms(const std::vector<int32> &in) { bclass2xforms_ = in; }

 private:
  int32 dim_;             ///< Dimension of feature vectors
  int32 num_xforms_;            ///< Number of transform matrices
  std::vector< Matrix<BaseFloat> > xform_matrices_;  ///< Transform matrices
  Vector<BaseFloat> logdet_;    ///< Log-determinants of the Jacobians
  bool valid_logdet_;           ///< Whether logdets are for current transforms
  /// For each baseclass index of which transform to use; -1 => no xform
  std::vector<int32> bclass2xforms_;

  void operator = (const RegtreeFmllrDiagGmm&);  // Disallow assignment operator
};

inline void RegtreeFmllrDiagGmm::GetXformMatrix(int32 xform_index,
                                              Matrix<BaseFloat> *out) const {
  if (xform_index >= num_xforms_) {
    KALDI_ERR << "Index (" << xform_index << ") out of range [0, "
        << num_xforms_ << "]";
  }
  out->Resize(dim_, dim_ + 1);
  out->CopyFromMat(xform_matrices_[xform_index], kNoTrans);
}

inline void RegtreeFmllrDiagGmm::SetParameters(const MatrixBase<BaseFloat> &mat,
                                        size_t regclass) {
  xform_matrices_[regclass].CopyFromMat(mat, kNoTrans);
  valid_logdet_ = false;
}

inline void RegtreeFmllrDiagGmm::GetLogDets(VectorBase<BaseFloat> *out) const {
  KALDI_ASSERT(valid_logdet_ && out->Dim() == logdet_.Dim());
  out->CopyFromVec(logdet_);
}

typedef TableWriter< KaldiObjectHolder<RegtreeFmllrDiagGmm> >  RegtreeFmllrDiagGmmWriter;
typedef RandomAccessTableReader< KaldiObjectHolder<RegtreeFmllrDiagGmm> >
            RandomAccessRegtreeFmllrDiagGmmReader;
typedef RandomAccessTableReaderMapped< KaldiObjectHolder<RegtreeFmllrDiagGmm> >
            RandomAccessRegtreeFmllrDiagGmmReaderMapped;
typedef SequentialTableReader< KaldiObjectHolder<RegtreeFmllrDiagGmm> >  RegtreeFmllrDiagGmmSeqReader;

/** \class RegtreeFmllrDiagGmmAccs
 *  Class for computing the accumulators needed for the maximum-likelihood
 *  estimate of FMLLR transforms for an acoustic model that uses diagonal
 *  Gaussian mixture models as emission densities.
 */
class RegtreeFmllrDiagGmmAccs {
 public:
  RegtreeFmllrDiagGmmAccs() : num_baseclasses_(-1), dim_(-1) {}
  ~RegtreeFmllrDiagGmmAccs() { DeletePointers(&baseclass_stats_); }

  void Init(size_t num_bclass, size_t dim);
  void SetZero();

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work if the features have already been transformed
  /// with multiple feature transforms (so you can't use use this to
  /// do a 2nd pass of regression-tree fMLLR estimation, which as I write
  /// (Dan, 2016) I'm not sure that this framework even supports.
  BaseFloat AccumulateForGmm(const RegressionTree &regtree,
                             const AmDiagGmm &am,
                             const VectorBase<BaseFloat> &data,
                             size_t pdf_index, BaseFloat weight);

  /// Accumulate stats for a single Gaussian component in the model.
  void AccumulateForGaussian(const RegressionTree &regtree,
                             const AmDiagGmm &am,
                             const VectorBase<BaseFloat> &data,
                             size_t pdf_index, size_t gauss_index,
                             BaseFloat weight);

  void Update(const RegressionTree &regtree, const RegtreeFmllrOptions &opts,
              RegtreeFmllrDiagGmm *out_fmllr, BaseFloat *auxf_impr,
              BaseFloat *tot_t) const;

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
  /// Number of baseclasses
  int32 num_baseclasses_;
  /// Dimension of feature vectors
  int32 dim_;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(RegtreeFmllrDiagGmmAccs);
};




}  // namespace kaldi

#endif  // KALDI_TRANSFORM_REGTREE_FMLLR_DIAG_GMM_H_

// transform/transform-common.h

// Copyright 2009-2011  Saarland University;  Georg Stemmer

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

#ifndef KALDI_TRANSFORM_TRANSFORM_COMMON_H_
#define KALDI_TRANSFORM_TRANSFORM_COMMON_H_

#include <vector>

#include "matrix/matrix-lib.h"

namespace kaldi {


class AffineXformStats {
 public:
  /// beta_ is the occupation count.
  double beta_;        
  /// K_ is the summed outer product of [mean times inverse variance] with [extended data],
  /// scaled by the occupation counts; dimension is dim by (dim+1)
  Matrix<double> K_;   
  /// G_ is the outer product of extended-data, scaled by inverse variance, for each
  /// dimension.  These are the quadratic stats in fMLLR; in the diagonal-fMLLR
  /// case G will be indexed 0 to dim_ - 1, but in the full-fMLLR case it will
  /// be indexed 0 to ((dim)(dim+1))/2.  Each G_[i] is of dimension dim+1 by dim+1.
  std::vector< SpMatrix<double> > G_;
  /// dim_ is the feature dimension.
  int32 dim_;      
  AffineXformStats(): beta_(0.0), dim_(0.0) {}
  void Init(int32 dim, int32 num_gs);  // num_gs will equal dim for diagonal FMLLR.
  int32 Dim() const { return dim_; }
  void SetZero();
  void CopyStats(const AffineXformStats &other);
  void Add(const AffineXformStats &other);
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary, bool add);
  AffineXformStats(const AffineXformStats &other): beta_(other.beta_),
                                                   K_(other.K_),
                                                   G_(other.G_),
                                                   dim_(other.dim_) {}
  // Note: allowing copy and assignment with their default
  // values.  All class members are OK with this.
};

bool ComposeTransforms(const Matrix<BaseFloat> &a, const Matrix<BaseFloat> &b,
                       bool b_is_affine,
                       Matrix<BaseFloat> *c);

/// Applies the affine transform 'xform' to the vector 'vec' and overwrites the
/// contents of 'vec'.
void ApplyAffineTransform(const MatrixBase<BaseFloat> &xform,
                          VectorBase<BaseFloat> *vec);



}  // namespace kaldi

#endif  // KALDI_TRANSFORM_TRANSFORM_COMMON_H_

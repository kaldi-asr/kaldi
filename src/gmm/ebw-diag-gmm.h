// gmm/ebw-diag-gmm.h

// Copyright 2009-2011  Arnab Ghoshal

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


#ifndef KALDI_GMM_EBW_DIAG_GMM_H_
#define KALDI_GMM_EBW_DIAG_GMM_H_ 1

#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/model-common.h"
#include "util/parse-options.h"

namespace kaldi {

class AccumEbwDiagGmm {
 public:
  AccumEbwDiagGmm(): dim_(0), num_comp_(0), flags_(0) {}
  explicit AccumEbwDiagGmm(const DiagGmm &gmm, GmmFlagsType flags) {
    Resize(gmm.NumGauss(), gmm.Dim(), flags);
  }
  // provide copy constructor.
  explicit AccumEbwDiagGmm(const AccumEbwDiagGmm &other);

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Allocates memory for accumulators
  void Resize(int32 num_comp, int32 dim, GmmFlagsType flags);

  /// Returns the number of mixture components
  int32 NumGauss() const { return num_comp_; }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return dim_; }

  void SetZero(GmmFlagsType flags);
  void Scale(BaseFloat f, GmmFlagsType flags);

  // TODO(arnab): maybe it's better to acc using a singe posterior, but we
  // need to know which occ stats to add to. Create 2 functions instead?
  /// Accumulate for all components, given the posteriors.
  void AccumulateFromPosteriors(const VectorBase<BaseFloat>& data,
                                const VectorBase<BaseFloat>& pos_posts,
                                const VectorBase<BaseFloat>& neg_posts);


  // TODO(arnab): we could keep the smoothing functions here as well. For
  // example, MPE stats will be directly accumulated as EBW stats and they
  // need to be smoothed. For MMIE, the numerator accumulator can be smoothed
  // before doing the subtraction.

  /// Smooths the accumulated counts using some other accumulator. Performs
  /// a weighted sum of the current accumulator with the given one. An example
  /// use for this is I-smoothing for MPE. Both accumulators must have the same
  /// dimension and number of components.
  void SmoothWithAccum(BaseFloat tau, const AccumDiagGmm& src_acc);

  /// Smooths the accumulated counts using the parameters of a given model.
  /// An example use of this is MAP-adaptation. The model must have the
  /// same dimension and number of components as the current accumulator.
  void SmoothWithModel(BaseFloat tau, const DiagGmm& src_gmm);

  // Accessors
  const GmmFlagsType Flags() const { return flags_; }
  const Vector<double>& num_occupancy() const { return num_occupancy_; }
  const Vector<double>& den_occupancy() const { return den_occupancy_; }
  const Matrix<double>& mean_accumulator() const { return mean_accumulator_; }
  const Matrix<double>& variance_accumulator() const { return variance_accumulator_; }

 private:
  int32 dim_;
  int32 num_comp_;
  /// Flags corresponding to the accumulators that are stored.
  GmmFlagsType flags_;

  Vector<double> num_occupancy_;
  Vector<double> den_occupancy_;
  Matrix<double> mean_accumulator_;
  Matrix<double> variance_accumulator_;
};

}  // End namespace kaldi


#endif  // KALDI_GMM_EBW_DIAG_GMM_H_

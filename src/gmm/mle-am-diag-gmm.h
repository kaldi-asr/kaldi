// gmm/mle-am-diag-gmm.h

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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


#ifndef KALDI_GMM_ESTIMATE_AM_DIAG_GMM_H_
#define KALDI_GMM_ESTIMATE_AM_DIAG_GMM_H_ 1

#include <vector>

#include "gmm/am-diag-gmm.h"
#include "gmm/mle-diag-gmm.h"

namespace kaldi {


/** \class MlEstimateAmDiagGmm
 *  Class for computing the maximum-likelihood estimates of the parameters of
 *  an acoustic model that uses diagonal Gaussian mixture models as emission
 *  densities.
 */
class MlEstimateAmDiagGmm {
 public:
  MlEstimateAmDiagGmm() { }
  ~MlEstimateAmDiagGmm();
  /// Initializes accumulators for each GMM based on the number of components
  /// and dimension.
  void InitAccumulators(const AmDiagGmm &model, GmmFlagsType flags);
  /// Initialization using different dimension than model.
  void InitAccumulators(const AmDiagGmm &model, int32 dim,
                        GmmFlagsType flags);
  void ZeroAccumulators(GmmFlagsType flags);

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work with multiple feature transforms.
  BaseFloat AccumulateForGmm(const AmDiagGmm &model,
                             const VectorBase<BaseFloat>& data,
                             int32 gmm_index, BaseFloat weight);

  /// Accumulate stats for a single GMM in the model; uses data1 for
  /// getting posteriors and data2 for stats. Returns log likelihood.
  BaseFloat AccumulateForGmmTwofeats(const AmDiagGmm &model,
                                     const VectorBase<BaseFloat>& data1,
                                     const VectorBase<BaseFloat>& data2,
                                     int32 gmm_index, BaseFloat weight);

  /// Accumulates stats for a single GMM in the model using pre-computed
  /// Gaussian posteriors.
  void AccumulateFromPosteriors(const AmDiagGmm &model,
                                const VectorBase<BaseFloat>& data,
                                int32 gmm_index,
                                const VectorBase<BaseFloat>& posteriors);

  /// Accumulate stats for a single Gaussian component in the model.
  void AccumulateForGaussian(const AmDiagGmm &am,
                             const VectorBase<BaseFloat>& data,
                             int32 gmm_index, int32 gauss_index,
                             BaseFloat weight);

  void Update(const MleDiagGmmOptions &config, GmmFlagsType flags,
              AmDiagGmm *am_gmm, BaseFloat *obj_change_out,
              BaseFloat *count_out) const;

  void Read(std::istream &in_stream, bool binary, bool add);

  void Write(std::ostream &out_stream, bool binary) const;

  MlEstimateDiagGmm& GetAcc(int32 index);

  const MlEstimateDiagGmm& GetAcc(int32 index) const;

  int32 NumAccs() { return gmm_estimators_.size(); }

 private:
  /// MLE accumulators and update methods for the GMMs
  std::vector<MlEstimateDiagGmm*> gmm_estimators_;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(MlEstimateAmDiagGmm);
};

}  // End namespace kaldi


#endif  // KALDI_GMM_ESTIMATE_AM_DIAG_GMM_H_

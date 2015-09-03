// gmm/mle-am-diag-gmm.h

// Copyright 2009-2012  Saarland University (author: Arnab Ghoshal);
//                      Yanmin Qian; Johns Hopkins University (author: Daniel Povey)
//                      Cisco Systems (author: Neha Agrawal)

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


#ifndef KALDI_GMM_MLE_AM_DIAG_GMM_H_
#define KALDI_GMM_MLE_AM_DIAG_GMM_H_ 1

#include <vector>

#include "gmm/am-diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "util/common-utils.h"

namespace kaldi {

class AccumAmDiagGmm {
 public:
  AccumAmDiagGmm() : total_frames_(0.0), total_log_like_(0.0) {}
  ~AccumAmDiagGmm();

  void Read(std::istream &in_stream, bool binary, bool add = false);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Initializes accumulators for each GMM based on the number of components
  /// and dimension.
  void Init(const AmDiagGmm &model, GmmFlagsType flags);
  /// Initialization using different dimension than model.
  void Init(const AmDiagGmm &model, int32 dim, GmmFlagsType flags);
  void SetZero(GmmFlagsType flags);

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work with multiple feature transforms.
  BaseFloat AccumulateForGmm(const AmDiagGmm &model,
                             const VectorBase<BaseFloat> &data,
                             int32 gmm_index, BaseFloat weight);

  /// Accumulate stats for a single GMM in the model; uses data1 for
  /// getting posteriors and data2 for stats. Returns log likelihood.
  BaseFloat AccumulateForGmmTwofeats(const AmDiagGmm &model,
                                     const VectorBase<BaseFloat> &data1,
                                     const VectorBase<BaseFloat> &data2,
                                     int32 gmm_index, BaseFloat weight);

  /// Accumulates stats for a single GMM in the model using pre-computed
  /// Gaussian posteriors.
  void AccumulateFromPosteriors(const AmDiagGmm &model,
                                const VectorBase<BaseFloat> &data,
                                int32 gmm_index,
                                const VectorBase<BaseFloat> &posteriors);

  /// Accumulate stats for a single Gaussian component in the model.
  void AccumulateForGaussian(const AmDiagGmm &am,
                             const VectorBase<BaseFloat> &data,
                             int32 gmm_index, int32 gauss_index,
                             BaseFloat weight);

  int32 NumAccs() { return gmm_accumulators_.size(); }

  int32 NumAccs() const { return gmm_accumulators_.size(); }

  BaseFloat TotStatsCount() const; // returns the total count got by summing the count
  // of the actual stats, may differ from TotCount() if e.g. you did I-smoothing.
  
  // Be careful since total_frames_ is not updated in AccumulateForGaussian
  BaseFloat TotCount() const { return total_frames_; }
  BaseFloat TotLogLike() const { return total_log_like_; }

  const AccumDiagGmm& GetAcc(int32 index) const;

  AccumDiagGmm& GetAcc(int32 index);

  void Add(BaseFloat scale, const AccumAmDiagGmm &other);

  void Scale(BaseFloat scale);

  int32 Dim() const {
    return (gmm_accumulators_.empty() || !gmm_accumulators_[0] ?
            0 : gmm_accumulators_[0]->Dim());
  }

 private:
  /// MLE accumulators and update methods for the GMMs
  std::vector<AccumDiagGmm*> gmm_accumulators_;

  /// Total counts & likelihood (for diagnostics)
  double total_frames_, total_log_like_;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(AccumAmDiagGmm);
};

/// for computing the maximum-likelihood estimates of the parameters of
/// an acoustic model that uses diagonal Gaussian mixture models as emission densities.
void MleAmDiagGmmUpdate(const MleDiagGmmOptions &config,
                        const AccumAmDiagGmm &amdiaggmm_acc,
                        GmmFlagsType flags,
                        AmDiagGmm *am_gmm,
                        BaseFloat *obj_change_out,
                        BaseFloat *count_out);

/// Maximum A Posteriori update.
void MapAmDiagGmmUpdate(const MapDiagGmmOptions &config,
                        const AccumAmDiagGmm &diag_gmm_acc,
                        GmmFlagsType flags,
                        AmDiagGmm *gmm,
                        BaseFloat *obj_change_out,
                        BaseFloat *count_out);

// These typedefs are needed to write GMMs to and from pipes, for MAP
// adaptation and decoding.  Note: this doesn't handle the transition
// model, you have to read that in separately.
typedef TableWriter< KaldiObjectHolder<AmDiagGmm> >  MapAmDiagGmmWriter;
typedef RandomAccessTableReader< KaldiObjectHolder<AmDiagGmm> > RandomAccessMapAmDiagGmmReader;
typedef RandomAccessTableReaderMapped< KaldiObjectHolder<AmDiagGmm> > RandomAccessMapAmDiagGmmReaderMapped;
typedef SequentialTableReader< KaldiObjectHolder<AmDiagGmm> > MapAmDiagGmmSeqReader;

}  // End namespace kaldi


#endif  // KALDI_GMM_MLE_AM_DIAG_GMM_H_

// ctc/cctc-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CTC_CCTC_TRAINING_H_
#define KALDI_CTC_CCTC_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "ctc/language-model.h"

namespace kaldi {
namespace ctc {

// CTC means Connectionist Temporal Classification, see the paper by Graves et
// al.  CCTC means context-dependent CTC, it's an extension of the original model,
// in which the next phone is dependent on the phone history (actually, a truncation
// thereof) in addition to the acoustic history.


struct CctcTrainingOptions {
  BaseFloat normalizing_weight;
  BaseFloat min_post;

  CctcTrainingOptions(): normalizing_weight(0.0001) { }

  void Register(OptionsItf *opts) {
    opts->Register("normalizing-weight", &normalizing_weight, "Weight on a "
                   "term in the objective function that's a negative squared "
                   "log of the numerator in the CCTC likelihood; it "
                   "exists to keep the network outputs in a reasonable "
                   "range so we can exp() them without overflow.");
  }
  
};


// This class is used while training CCTC models and evaluating probabilities on
// held-out training data.  It is not responsible for the entire process of CCTC
// model training; it is only responsible for the forward-backward from the
// neural net output, and the derivative computation.

class CctcComputation {
 public:
  CctcComputation(const CctcTrainingOptions &opts,
                  const CctcTransitionModel &trans_model,
                  const CuMatrix<BaseFloat> &cu_weights,
                  const CtcSupervision &supervision,
                  const CuMatrixBase<BaseFloat> &nnet_output);

  // Does the forward computation.  Returns the total log-prob.
  BaseFloat Forward();
                    
  // Does the backward computation and adds the derivative w.r.t. the neural
  // network output to 'nnet_output_deriv' (so you should probably set it to
  // zero beforehand).
  // Returns true if everything was OK (which it should be, normally), and
  // false if some kind of NaN or inf was discovered, in which case you
  // shouldn't use the derivatives.  We're concerned about this because
  // the setup takes exponentials of neural network outputs without applying
  // any ceiling.
  bool Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);

 private:

  const CctcTrainingOptions &opts_;
  const CctcTransitionModel &trans_model_;
  // CUDA copy of trans_model_.Weights().  Dimension is
  // trans_model_.NumHistoryStates() by trans_model_.NumOutputIndexes().
  const CuMatrix<BaseFloat> &weights_;

  const CtcSupervision &supervision_;
  
  // The neural net output
  const CuMatrixBase<BaseFloat> &nnet_output_;
  // the exp of the neural net output.
  CuMatrix<BaseFloat> exp_nnet_output_;
  // the normalizers (denominator terms), of dimension nnet_output_.NumRows() by
  // trans_model_.NumHistoryStates(), equal to exp_nnet_output_ * weights_'.
  CuMatrix<BaseFloat> normalizers_;


  // 'fst_indexes' contains an entry for each arc in the supervision FST, in
  // the order you'd get them if you visit each arc of each state in order.
  // The first member of each pair is the index into the
  // numerator_indexes/numerator_probs saying where we need to look up the
  // numerator likelihood; the second is the index into
  // denominator_indexes/denominator_probs, saying where we need to look up
  // the denominator likelihood.
  std::vector<std::pair<int32,int32> > fst_indexes_;
  // This has the same number of elements as fst_indexes_; it's used to store
  // the probabilities on each arc of the FST (these are of the form
  // numerator-prob * lm_prob / denominator-prob), where lm_prob is the
  // phone-language-model probability, taken from the transition model.
  Vector<BaseFloat> arc_probs_;
  
  // numerator_indexes is a list of indexes that we need to look up in
  // exp_nnet_output_ for the forward-backward computation.  The order is not
  // important, but indexes into this vector appear in .first members in
  // fst_indexes.
  std::vector<Int32Pair> numerator_indexes_;
  // the numerator of the probability.  in the forward computation,
  // numerator_probs_[i] equals exp_nnet_output_(row,column), where (row,column)
  // is the i'th member of numerator_indexes.  In the backward computation,
  // the storage is reused for derivatives.
  Vector<BaseFloat> numerator_probs_;
    

  // denominator_indexes is a list of indexes that we need to look up in
  // normalizers_ for the forward-backward computation.  The order is not
  // important, but indexes into this vector appear in .second members in
  // fst_indexes.
  std::vector<Int32Pair> denominator_indexes_;
  // the denominator of the probability.  denominator_probs_[i] equals
  // exp_nnet_output_(row,column), where (row,column) is the i'th member of
  // denominator_indexes.
  Vector<BaseFloat> denominator_probs_;

  // This quantity, with the same dimension as denominator_probs_, is used in
  // the backward computation to store derivatives w.r.t. the denominator
  // values.
  Vector<BaseFloat> denominator_deriv_;

  // The log-alpha value (forward probability) for each state in the lattice
  Vector<double> alpha_;

  // The total log-probability of the supervision (you can interpret this as
  // the posterior of this phone-sequence).
  double tot_log_prob_;

  // The log-beta value (backward probability) for each state in the lattice  
  Vector<double> beta_;

 private:
  // This function, called from the constructor, checks various dimensions.
  void CheckDims() const;
  
  //  This function, called from Forward(), creates fst_indexes_,
  //  numerator_indexes_ and denominator_indexes_.
  void ComputeLookupIndexes();

  // This function, called from Forward(), computes denomator_probs_ and
  // numerator_probs_ via batch lookup operations in exp_nnet_output_ and
  // normalizers_, and then computes arc_probs_.
  void LookUpLikelihoods();

  // This function, called from Forward(), does the actual forward-computation on
  // the FST, setting alpha_ and tot_log_prob_.
  void ComputeAlpha();

  // Computes the beta probabilities (called from Backward().)
  void ComputeBeta();

  // Computes derivatives (called from Backward()).
  // Returns true on success, false if a NaN or Inf was detected.
  bool ComputeDerivatives(CuMatrixBase<BaseFloat> *nnet_output_deriv);
  
  
};



}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_TRAINING_H_


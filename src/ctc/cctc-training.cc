// ctc/cctc-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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


#include "ctc/cctc-training.h"

namespace kaldi {
namespace ctc {

void CctcTrainer::CheckDims() const {
  KALDI_ASSERT(weights_.NumRows() == trans_model_.NumHistoryStates() &&
               weights_.NumCols() == trans_model_.NumIndexes());
  KALDI_ASSERT(nnet_output_.NumRows() == supervision_.num_frames &&
               nnet_output_.NumCols() == trans_model_.NumIndexes());
  KALDI_ASSERT(supervision_.label_dim == trans_model_.NumIndexes());
}

void CctcTraining::Forward() {
  CheckDims();
  ComputeLookupIndexes();
  exp_nnet_output_ = nnet_output_;
  exp_nnet_output_.ApplyExp();
  normalizers_.Resize(exp_nnet_output_.NumRows(),
                      trans_model_.NumHistoryStates());
  normalizers_.AddMatMat(1.0, exp_nnet_output_, kNoTrans, weights_, kTrans);
  LookUpLikelihoods();
  ComputeAlphas();
}



bool CctcTraining::Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  ComputeBeta();
  return ComputeDerivatives(nnet_output_deriv);
}
  

bool CctcTraining::ComputeDerivatives(
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  // we assume nnet_output_deriv is already zeroed; we add to it.
  int32 num_states = supervision_.fst.NumStates();
  int32 arc_index = 0;  // Index of arc in global tables of arcs.
  const double *alpha_data = alpha_.Data(),
      *beta_data =  beta_.Data();

  std::vector<std::pair<int32,int32> >::const_iterator fst_indexes_iter =
      fst_indexes_.begin();

  numerator_probs_.SetZero();  // we'll use this to store derivatives w.r.t. the
                               // numerator log-prob; these derivatives are just
                               // sums of occupation counts.
  BaseFloat numerator_deriv_data = numerator_probs_.Data();

  // size and zero denominator_deriv_.  It will contain the sum of negated
  // occupancies that map to each element of the denominator_indexes_ and
  // denominator_prob_ vectors.
  denominator_deriv_.Resize(denominator_probs_.Dim());
  BaseFloat denominator_deriv_data = denominator_deriv_.Data();
  
  const BaseFloat *arc_prob_data = arc_probs_.Data();
  for (int32 state = 0; state < num_states; state++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(supervision_.fst, state);
         !aiter.Done(); aiter.Next(), ++arc_index, ++fst_indexes_iter) {
      const fst::StdArc &arc = aiter.Value();
      int32 nextstate = arc.nextstate;
      double arc_posterior =
          exp(alpha_data[state] + beta_data[nextstate] - tot_log_prob_) *
          arc_probs_[arc_index];
      KALDI_ASSERT(arc_prob >= 0.0 && arc_prob < 1.1);
      int32 numerator_index = fst_indexes_iter->first,
          denominator_index = fst_indexes_iter->second;
      // interpret this as d(objf)/d(log of numerator)
      numerator_deriv_data[numerator_index] += arc_posterior;
      // interpret this at this point, as d(objf)/d(log of denominator)
      denominator_deriv_data[denominator_index] -= arc_posterior;
    }
  }
  // Change denominator_deriv_ from being d(objf)/d(log denominator)
  // to being d(objf)/d(denominator).  This division is why we couldn't reuse
  // denominator_probs_ itself as the derivative.
  denominator_deriv_.DivElements(denominator_probs_);

  // We will reuse the normalizers_ array to be the derivatives
  // w.r.t. the normalizers.
  normalizers_.SetZero();
  
  

}



  // Computing the alphas:

  // for each  arc, compute lm_prob * num / den.
  // alpha is as in regular algorithm.

  // compute the total forward prob.
  // Computing the betas:
  // for each state backwards
  //    - compute its beta using previously computed arc probs.
  //    - compute arc posterior = (1/total_forward_prob) * alpha[start] * prob[arc] * beta[end].
  //     [we will compute arc-posterior sums as a check.]
  //    -  Given the arc posterior, can compute d(objf)/d[arc-posterior].  If objf == logprob,
  //    - d(logprob) / d(log arc) = arc posterior.
  //    - d(logprob) / d(log numerator) = arc posterior.
  //    - d(logprob) /d(log denominator) = -arc posterior.
  //
  // the nnet outputs the logprob directly.  We just need to add +(arc posterior) to the
  //          numerator positions.
  //
  // For the denominator probs, we need to backprop through the linear matrix, which requires
  // going to non-log-space.  Compute d(logprob)/d(denominator) = -arc-posterior / denominator.
  // Then compute the denominator part of d(logprob)/d(whole-exped-matrix) by multiplying by
  // the weights matrix.  Then multiply by exped-matrix to get d(logprob)/d(whole-orig-matrix).
  // then add d(logprob)/d(log-numerator).

  
  
  // we have the numerator and denominator values.
  
  // lm_prob * num / den.
  
}


}  // namespace ctc
}  // namespace kaldi

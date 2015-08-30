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


CctcComputation::CctcComputation(
    const CctcTrainingOptions &opts,
    const CctcTransitionModel &trans_model,
    const CuMatrix<BaseFloat> &cu_weights,
    const CtcSupervision &supervision,
    const CuMatrixBase<BaseFloat> &nnet_output):
    opts_(opts), trans_model_(trans_model), cu_weights_(cu_weights),
    supervision_(supervision), nnet_output_(nnet_output) {
  CheckDims();
}


void CctcComputation::CheckDims() const {
  KALDI_ASSERT(cu_weights_.NumRows() == trans_model_.NumHistoryStates() &&
               cu_weights_.NumCols() == trans_model_.NumOutputIndexes());
  KALDI_ASSERT(nnet_output_.NumRows() == supervision_.num_frames &&
               nnet_output_.NumCols() == trans_model_.NumOutputIndexes());
  KALDI_ASSERT(supervision_.label_dim == trans_model_.NumOutputIndexes());
}

// This function, called from Forward(), does the actual forward-computation on
// the FST, setting alpha_ and tot_log_prob_.
// Note: the FST is unweighted.
void CctcComputation::ComputeAlpha() {
  const fst::StdVectorFst &fst = supervision_.fst;
  KALDI_ASSERT(fst.Start() == 0);
  int32 num_states = fst.NumStates();
  log_alpha_.Resize(num_states, kUndefined);
  log_alpha_.Set(-std::numeric_limits<double>::infinity());
  tot_log_prob_ = -std::numeric_limits<double>::infinity();

  log_alpha_(0) = 0.0;  // it's in log space.
  int32 arc_index = 0;  // arc_index will index fst_indexes_.
  const BaseFloat *arc_logprob_data = &(arc_logprobs_[0]);

  for (int32 state = 0; state < num_states; state++) {
    double this_alpha = log_alpha_(state);
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state); !aiter.Done();
         aiter.Next(), arc_index++) {
      int32 nextstate = aiter.Value().nextstate;
      double arc_logprob = arc_logprob_data[arc_index];
      double &next_alpha = log_alpha_(nextstate);
      next_alpha = LogAdd(next_alpha, arc_logprob + this_alpha);
    }
    if (fst.Final(state) != fst::TropicalWeight::Zero())
      tot_log_prob_ = LogAdd(tot_log_prob_, this_alpha);
  }
  KALDI_ASSERT(arc_index == static_cast<int32>(arc_logprobs_.size()));
}

void CctcComputation::ComputeBeta() {
  const fst::StdVectorFst &fst = supervision_.fst;
  int32 num_states = fst.NumStates();
  log_beta_.Resize(num_states, kUndefined);

  // we'll be counting backwards and moving the 'arc_logprob_iter' pointer back.
  const BaseFloat *arc_logprob_data = &(arc_logprobs_[0]),
      *arc_logprob_iter = arc_logprob_data + arc_logprobs_.size();

  for (int32 state = num_states - 1; state >= 0; state--) {
    int32 this_num_arcs  = fst.NumArcs(state);
    // on the backward pass we access the arc_logprobs_ vector in a zigzag
    // pattern.

    arc_logprob_iter -= this_num_arcs;
    const BaseFloat *this_arc_logprob_iter = arc_logprob_iter;
    double this_beta = (fst.Final(state) == fst::TropicalWeight::Zero() ?
                        -std::numeric_limits<double>::infinity() : 0.0);
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state); !aiter.Done();
         aiter.Next(), this_arc_logprob_iter++) {
      double arc_logprob = *this_arc_logprob_iter;
      double next_beta = arc_logprob_data[aiter.Value().nextstate];
      this_beta = LogAdd(this_beta, arc_logprob + next_beta);

    }
    KALDI_PARANOID_ASSERT(this_beta != -std::numeric_limits<double>::infinity());
    log_beta_(state) = this_beta;
  }    
  KALDI_ASSERT(arc_logprob_iter == arc_logprob_data);

  int32 start_state = 0;  // We alredy checked this.
  double tot_log_prob_backward = log_beta_(start_state);
  if (!ApproxEqual(tot_log_prob_backward, tot_log_prob_))
    KALDI_WARN << "Disagreement in forward/backward log-probs: "
               << tot_log_prob_backward << " vs. " << tot_log_prob_;
  
}


void CctcComputation::ComputeLookupIndexes() {
  std::vector<int32> fst_state_times;
  ComputeFstStateTimes(supervision_.fst, &fst_state_times);
  int32 num_states = supervision_.fst.NumStates();
  int32 num_arcs_guess = num_states * 2;
  fst_indexes_.reserve(num_arcs_guess);
  arc_logprobs_.reserve(num_arcs_guess);
  int32 cur_time = 0;

  // the following are CPU versions of numerator_indexes_ and
  // denominator_indexes_.  numerator_indexes_cpu is a list of pairs (t,
  // output-index) and denominator_indexes_cpu is a list of pairs (c,
  // history-state-index).
  std::vector<Int32Pair> numerator_indexes_cpu, denominator_indexes_cpu;
  // numerator_index_map_this_frame is a map, only valid for t == cur_time,
  // from the output-index to the index into numerator_indexes_cpu for the
  // p air (cur_time, output-index).
  unordered_map<int32,int32> numerator_index_map_this_frame;
  // denoninator_index_map_this_frame is a map, only valid for t == cur_time,
  // from the output-index to the index into numerator_indexes_cpu for the
  // p air (cur_time, output-index).
  unordered_map<int32,int32> denominator_index_map_this_frame;

  typedef unordered_map<int32,int32>::iterator IterType;
  
  for (int32 state = 0; state < num_states; state++) {
    int32 t = fst_state_times[state];
    if (t != cur_time) {
      KALDI_ASSERT(t == cur_time + 1);
      numerator_index_map_this_frame.clear();
      denominator_index_map_this_frame.clear();
      cur_time = t;
    }
    for (fst::ArcIterator<fst::StdVectorFst> aiter(supervision_.fst, state);
         !aiter.Done(); aiter.Next()) {
      int32 graph_label = aiter.Value().ilabel,
          output_index = trans_model_.GraphLabelToOutputIndex(graph_label),
          history_state = trans_model_.GraphLabelToHistoryState(graph_label);

      int32 numerator_index = numerator_indexes_cpu.size(),
          denominator_index = denominator_indexes_cpu.size();
      Int32Pair num_pair, den_pair;  // can't use constructors as declared in C.
      num_pair.first = t;
      num_pair.second = output_index;
      den_pair.first = t;
      den_pair.second = history_state;
      // the next few lines are a more efficient way of doing the following:
      // if (numerator_index_map_this_frame.count(output_index) == 0) {
      //   numerator_index_map_this_frame[output_index] = numerator_index;
      // else
      //   numerator_index = numerator_index_map_this_frame[output_index];
      std::pair<IterType,bool> p = numerator_index_map_this_frame.insert(
              std::pair<const int32, int32>(output_index, numerator_index));
      if (p.second) {  // Was inserted -> map had no key 'output_index'
        numerator_indexes_cpu.push_back(num_pair);
      } else {  // was not inserted -> set numerator_index to the existing index.
        numerator_index = p.first->second;
        KALDI_PARANOID_ASSERT(numerator_indexes_cpu[numerator_index] ==
                              num_pair);
      }
      // the next few lines are a more efficient way of doing the following:
      // if (denominator_index_map_this_frame.count(history_state) == 0) {
      //   denominator_index_map_this_frame[history_state] = denominator_index;
      // else
      //   denominator_index = denominator_index_map_this_frame[history_state];
      p = denominator_index_map_this_frame.insert(
          std::pair<const int32, int32>(history_state, denominator_index));
      if (p.second) {  // Was inserted -> map had no key 'history_state'
        denominator_indexes_cpu.push_back(den_pair);
      } else {  // was not inserted -> set denominator_index to the existing index.
        denominator_index = p.first->second;
        KALDI_PARANOID_ASSERT(denominator_indexes_cpu[denominator_index] ==
                               den_pair);
      }
      fst_indexes_.push_back(std::pair<int32,int32>(numerator_index,
                                                    denominator_index));
      arc_logprobs_.push_back(trans_model_.GraphLabelToLmProb(graph_label));
    }
  }
  numerator_indexes_ = numerator_indexes_cpu;
  denominator_indexes_ = denominator_indexes_cpu;
  KALDI_ASSERT(!fst_indexes_.empty());
}

BaseFloat CctcComputation::Forward() {
  ComputeLookupIndexes();
  exp_nnet_output_ = nnet_output_;
  exp_nnet_output_.ApplyExp();
  normalizers_.Resize(exp_nnet_output_.NumRows(),
                      trans_model_.NumHistoryStates());
  normalizers_.AddMatMat(1.0, exp_nnet_output_, kNoTrans, cu_weights_, kTrans,
                         0.0);
  LookUpLikelihoods();
  ComputeAlpha();
  return tot_log_prob_;
}

void CctcComputation::LookUpLikelihoods() {
  numerator_probs_.Resize(numerator_indexes_.Dim(), kUndefined);
  exp_nnet_output_.Lookup(numerator_indexes_, numerator_probs_.Data());
  denominator_probs_.Resize(denominator_indexes_.Dim(), kUndefined);
  normalizers_.Lookup(denominator_indexes_, denominator_probs_.Data());
  // Note: at this point, arc_logprobs_ contains the phone language model
  // probabilities.
  BaseFloat *arc_logprob_data = &(arc_logprobs_[0]);
  const BaseFloat *numerator_prob_data = numerator_probs_.Data(),
      *denominator_prob_data = denominator_probs_.Data();
  std::vector<std::pair<int32,int32> >::const_iterator
      iter = fst_indexes_.begin(), end = fst_indexes_.end();
  for (; iter != end; ++iter, ++arc_logprob_data)
    *arc_logprob_data = Log(*arc_logprob_data) *
        numerator_prob_data[iter->first] /
        denominator_prob_data[iter->second];
}

bool CctcComputation::Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  ComputeBeta();
  return ComputeDerivatives(nnet_output_deriv);
}
  

bool CctcComputation::ComputeDerivatives(
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  // we assume nnet_output_deriv is already zeroed; we add to it.
  int32 num_states = supervision_.fst.NumStates();
  int32 arc_index = 0;  // Index of arc in global tables of arcs.
  const double *log_alpha_data = log_alpha_.Data(),
      *log_beta_data =  log_beta_.Data();

  std::vector<std::pair<int32,int32> >::const_iterator fst_indexes_iter =
      fst_indexes_.begin();

  numerator_probs_.SetZero();  // we'll use this to store derivatives w.r.t. the
                               // numerator log-prob; these derivatives are just
                               // sums of occupation counts.
  BaseFloat *numerator_deriv_data = numerator_probs_.Data();

  // size and zero denominator_deriv_.  It will contain the sum of negated
  // occupancies that map to each element of the denominator_indexes_ and
  // denominator_prob_ vectors.
  denominator_deriv_.Resize(denominator_probs_.Dim());
  BaseFloat *denominator_deriv_data = denominator_deriv_.Data();
  
  const BaseFloat *arc_logprob_data = &(arc_logprobs_[0]);
  for (int32 state = 0; state < num_states; state++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(supervision_.fst, state);
         !aiter.Done(); aiter.Next(), ++arc_index, ++fst_indexes_iter) {
      const fst::StdArc &arc = aiter.Value();
      int32 nextstate = arc.nextstate;
      double arc_posterior =
          exp(log_alpha_data[state] + log_beta_data[nextstate] - tot_log_prob_) *
          arc_logprob_data[arc_index];
      KALDI_ASSERT(arc_posterior >= 0.0 && arc_posterior < 1.1);
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

  normalizers_.AddElements(1.0, denominator_indexes_,
                           denominator_deriv_data);

  // Even though the next statement adds it with zero coefficient, we need
  // to set it to zero to guard against inf's or NaN's.
  nnet_output_deriv->SetZero();
  
  // After the following statement, 'nnet_output_deriv' contains the derivative
  // with respect to 'exp_nnet_output_', considering only the denominator term.
  nnet_output_deriv->AddMatMat(1.0, normalizers_, kNoTrans,
                               cu_weights_, kNoTrans, 0.0);
  // After the following statement, 'nnet_output_deriv' contains the derivative with
  // respect to 'nnet_output_', considering only the denominator term.
  // we use that y/d(exp x) = exp(x) dy/dx.
  nnet_output_deriv->MulElements(exp_nnet_output_);

  // After the following statement, 'nnet_output_deriv' should contain the
  // entire derivative, also including the numerator term.  Note: at this point,
  // numerator_probs_ contains summed posteriors, which equal the derivative of
  // the likelihood w.r.t. the nnet log output (considering only the numerator
  // term).
  nnet_output_deriv->AddElements(1.0, numerator_indexes_, numerator_probs_.Data());

  BaseFloat sum = nnet_output_deriv->Sum();
  return (sum == sum && sum - sum == 0);  // check for NaN/inf.
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
  

}  // namespace ctc
}  // namespace kaldi

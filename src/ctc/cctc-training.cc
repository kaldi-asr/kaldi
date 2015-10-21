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


CctcPositiveComputation::CctcPositiveComputation(
    const CctcTrainingOptions &opts,
    const CctcTransitionModel &trans_model,
    const CctcSupervision &supervision,
    int32 num_sequences,
    const CuMatrixBase<BaseFloat> &exp_nnet_output,
    const CuMatrixBase<BaseFloat> &denominators):
    opts_(opts), trans_model_(trans_model),
    supervision_(supervision),
    num_sequences_(num_sequences),
    exp_nnet_output_(exp_nnet_output),
    denominators_(denominators) { }


// This function, called from Forward(), does the actual forward-computation on
// the FST, setting alpha_ and tot_log_prob_.
// Note: the FST is unweighted.
void CctcPositiveComputation::ComputeAlpha() {
  const fst::StdVectorFst &fst = supervision_.fst;
  KALDI_ASSERT(fst.Start() == 0);
  int32 num_states = fst.NumStates();
  log_alpha_.Resize(num_states, kUndefined);
  log_alpha_.Set(-std::numeric_limits<double>::infinity());
  tot_log_prob_ = -std::numeric_limits<double>::infinity();

  log_alpha_(0) = 0.0;  // note, state zero is the start state, we checked above
  const BaseFloat *arc_logprob_iter = &(arc_logprobs_[0]);
  double *log_alpha_data = log_alpha_.Data();

  for (int32 state = 0; state < num_states; state++) {
    double this_alpha = log_alpha_data[state];
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state); !aiter.Done();
         aiter.Next(), ++arc_logprob_iter) {
      int32 nextstate = aiter.Value().nextstate;
      double arc_logprob = *arc_logprob_iter;
      double &next_alpha = log_alpha_data[nextstate];
      next_alpha = LogAdd(next_alpha, arc_logprob + this_alpha);
    }
    if (fst.Final(state) != fst::TropicalWeight::Zero())
      tot_log_prob_ = LogAdd(tot_log_prob_, this_alpha);
  }
  KALDI_ASSERT(arc_logprob_iter == &(arc_logprobs_[0]) + arc_logprobs_.size());
}

void CctcPositiveComputation::ComputeBeta() {
  const fst::StdVectorFst &fst = supervision_.fst;
  int32 num_states = fst.NumStates();
  log_beta_.Resize(num_states, kUndefined);

  // we'll be counting backwards and moving the 'arc_logprob_iter' pointer back.
  const BaseFloat *arc_logprob_data = &(arc_logprobs_[0]),
      *arc_logprob_iter = arc_logprob_data + arc_logprobs_.size();
  double *log_beta_data = log_beta_.Data();

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
      double next_beta = log_beta_data[aiter.Value().nextstate];
      this_beta = LogAdd(this_beta, arc_logprob + next_beta);
    }
    KALDI_PARANOID_ASSERT(this_beta != -std::numeric_limits<double>::infinity());
    log_beta_data[state] = this_beta;
  }
  KALDI_ASSERT(arc_logprob_iter == arc_logprob_data);

  int32 start_state = 0;  // We alredy checked this.
  double tot_log_prob_backward = log_beta_(start_state);
  if (!ApproxEqual(tot_log_prob_backward, tot_log_prob_))
    KALDI_WARN << "Disagreement in forward/backward log-probs: "
               << tot_log_prob_backward << " vs. " << tot_log_prob_;

}


void CctcPositiveComputation::OutputFirstFrameAlpha(
    const std::vector<int32> &fst_state_times,
    CuVectorBase<BaseFloat> *first_frame_alpha) {
  std::vector<MatrixElement<BaseFloat> > nonzero_alphas;
  if (first_frame_alpha != NULL)
    nonzero_alphas.reserve(num_sequences_ * 4);  // a guess.
  KALDI_ASSERT(!fst_state_times.empty());
  // note: the FST should have an end-state, its time will equal the total
  // num-frames.
  int32 num_frames = fst_state_times.back();
  if (num_frames % num_sequences_ != 0)
    KALDI_ERR << "num-frames " << num_frames << " in supervision is not divided "
              << "by num-sequences " << num_sequences_;
  int32 frames_per_sequence = num_frames / num_sequences_;
  int32 num_sequence_initial_states = 0;
  std::vector<int32>::const_iterator begin = fst_state_times.begin(),
      iter = begin, end = fst_state_times.end();
  extra_log_prob_ = 0.0;
  for (; iter != end; ++iter) {
    int32 t = *iter;  // the time-index.
    if (t % frames_per_sequence == 0 && t != num_frames) {
      // this state's time divides the frames_per_sequence -> it's a
      // reconnection point.
      num_sequence_initial_states++;
      int32 state = iter - begin;  // The FST state.
      std::set<int32> history_states;
      fst::ArcIterator<fst::StdVectorFst> aiter(supervision_.fst, state);
      for (; !aiter.Done(); aiter.Next()) {
        int32 graph_label = aiter.Value().ilabel;
        int32 this_history_state = trans_model_.GraphLabelToHistoryState(
            graph_label);
        history_states.insert(this_history_state);
      }
      KALDI_ASSERT(!history_states.empty());
      extra_log_prob_ -= log(history_states.size());
      if (first_frame_alpha != NULL) {
        int32 sequence_index = t / frames_per_sequence;
        // the concept is that we divide the initial-prob evenly among
        // these history-states.
        BaseFloat prob = 1.0 / history_states.size();
        for (std::set<int32>::iterator set_iter = history_states.begin();
             set_iter != history_states.end(); ++set_iter) {
          int32 history_state = *set_iter;
          MatrixElement<BaseFloat> elem;
          elem.row = history_state;
          elem.column = sequence_index;
          elem.weight= prob;
          nonzero_alphas.push_back(elem);
        }
      }
    }
  }
  if (first_frame_alpha != NULL) {
    KALDI_ASSERT(first_frame_alpha->Dim() == num_sequences_ *
                 trans_model_.NumHistoryStates());
    // construct a fake matrix pointing to this data, with num-rows ==
    // num-hmm-states and num-cols == num-sequences.
    int32 num_hmm_states = trans_model_.NumHistoryStates();
    first_frame_alpha->SetZero();
    CuSubMatrix<BaseFloat> mat(first_frame_alpha->Data(),
                               num_hmm_states, num_sequences_,
                               num_sequences_);
    mat.AddElements(1.0, nonzero_alphas);
  }
}

void CctcPositiveComputation::ComputeLookupIndexes(
    CuVectorBase<BaseFloat> *first_frame_alpha) {
  std::vector<int32> fst_state_times;
  ComputeFstStateTimes(supervision_.fst, &fst_state_times);
  OutputFirstFrameAlpha(fst_state_times, first_frame_alpha);
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
      Int32Pair num_pair, den_pair;  // we can't use constructors as this was
                                     // declared in C.
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
      // Note: eventually arc_logprobs_ will contain the log of all the terms
      // in the arc probability, but at this point it's not a log.
      arc_logprobs_.push_back(trans_model_.GraphLabelToLmProb(graph_label));
    }
  }
  numerator_indexes_ = numerator_indexes_cpu;
  denominator_indexes_ = denominator_indexes_cpu;
  KALDI_ASSERT(!fst_indexes_.empty());
}

BaseFloat CctcPositiveComputation::Forward(
    CuVectorBase<BaseFloat> *first_frame_alpha) {
  ComputeLookupIndexes(first_frame_alpha);
  LookUpLikelihoods();
  ComputeAlpha();
  return tot_log_prob_ + extra_log_prob_;
}

void CctcPositiveComputation::LookUpLikelihoods() {
  numerator_probs_.Resize(numerator_indexes_.Dim(), kUndefined);
  exp_nnet_output_.Lookup(numerator_indexes_, numerator_probs_.Data());
  denominator_probs_.Resize(denominator_indexes_.Dim(), kUndefined);
  denominators_.Lookup(denominator_indexes_, denominator_probs_.Data());
  // Note: at this point, arc_logprobs_ contains the phone language model
  // probabilities.
  BaseFloat *arc_logprob_data = &(arc_logprobs_[0]);
  const BaseFloat *numerator_prob_data = numerator_probs_.Data(),
      *denominator_prob_data = denominator_probs_.Data();
  std::vector<std::pair<int32,int32> >::const_iterator
      iter = fst_indexes_.begin(), end = fst_indexes_.end();
  for (; iter != end; ++iter, ++arc_logprob_data) {
    *arc_logprob_data = Log(*arc_logprob_data *
                            numerator_prob_data[iter->first] /
                            denominator_prob_data[iter->second]);
    KALDI_PARANOID_ASSERT(*arc_logprob_data < 0.001); // should not be positive.
  }
}

void CctcPositiveComputation::Backward(
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    CuMatrixBase<BaseFloat> *denominators_deriv) {
  ComputeBeta();
  ComputeDerivatives(nnet_output_deriv, denominators_deriv);
}


void CctcPositiveComputation::ComputeDerivatives(
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    CuMatrixBase<BaseFloat> *denominators_deriv) {
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
          exp(log_alpha_data[state] + log_beta_data[nextstate] - tot_log_prob_ +
              arc_logprob_data[arc_index]);
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
  // to being d(objf)/d(denominator).
  denominator_deriv_.DivElements(denominator_probs_);

  denominators_deriv->AddElements(1.0, denominator_indexes_,
                                 denominator_deriv_data);

  // After the following statement, 'nnet_output_deriv' should contain the
  // numerator term of the derivative.  At this point, numerator_probs_ contains
  // summed posteriors, which equal the derivative of the likelihood w.r.t. the
  // nnet log output (considering only the numerator term).
  nnet_output_deriv->AddElements(1.0, numerator_indexes_, numerator_probs_.Data());

}


CctcCommonComputation::CctcCommonComputation(
    const CctcTrainingOptions &opts,
    const CctcTransitionModel &trans_model,
    const CuMatrix<BaseFloat> &cu_weights,
    const CctcSupervision &supervision,
    int32 num_sequences,
    const CuMatrixBase<BaseFloat> &nnet_output):
    hmm_(trans_model), opts_(opts), trans_model_(trans_model),
    cu_weights_(cu_weights),
    first_frame_alphas_(trans_model.NumHistoryStates() * num_sequences,
                        kUndefined),
    supervision_(supervision),
    num_sequences_(num_sequences), nnet_output_(nnet_output),
    positive_computation_(NULL), negative_computation_(NULL) {
  CheckDims();
}


void CctcCommonComputation::CheckDims() const {
  KALDI_ASSERT(cu_weights_.NumRows() == trans_model_.NumHistoryStates() &&
               cu_weights_.NumCols() == trans_model_.NumOutputIndexes());
  KALDI_ASSERT(nnet_output_.NumRows() == supervision_.num_frames &&
               nnet_output_.NumCols() == trans_model_.NumOutputIndexes());
  KALDI_ASSERT(supervision_.label_dim == trans_model_.NumGraphLabels());
}


void CctcCommonComputation::Forward(BaseFloat *positive_objf_part,
                                    BaseFloat *negative_objf_part,
                                    BaseFloat *objf_denominator) {
  exp_nnet_output_ = nnet_output_;
  exp_nnet_output_.ApplyExp();
  denominators_.Resize(exp_nnet_output_.NumRows(),
                      trans_model_.NumHistoryStates());
  denominators_.AddMatMat(1.0, exp_nnet_output_, kNoTrans, cu_weights_, kTrans,
                         0.0);
  denominators_deriv_.Resize(denominators_.NumRows(), denominators_.NumCols(),
                             kUndefined);

  KALDI_ASSERT(positive_computation_ == NULL && "Forward() called twice?");
  positive_computation_ = new CctcPositiveComputation(opts_, trans_model_,
                                                      supervision_,
                                                      num_sequences_,
                                                      exp_nnet_output_,
                                                      denominators_);

  *positive_objf_part = supervision_.weight *
      positive_computation_->Forward(&first_frame_alphas_);

  negative_computation_ = new CctcNegativeComputation(trans_model_, hmm_,
                                                      exp_nnet_output_,
                                                      denominators_,
                                                      num_sequences_,
                                                      &first_frame_alphas_);

  *negative_objf_part = -opts_.denominator_scale * supervision_.weight *
      negative_computation_->Forward();

  *objf_denominator = supervision_.weight * nnet_output_.NumRows();
}


CctcCommonComputation::~CctcCommonComputation() {
  delete positive_computation_;
  delete negative_computation_;
}

void CctcCommonComputation::Backward(
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  KALDI_ASSERT(SameDim(*nnet_output_deriv, nnet_output_));
  nnet_output_deriv->SetZero();

  // this function *sets* its output.
  negative_computation_->Backward(nnet_output_deriv,
                                  &denominators_deriv_);
  nnet_output_deriv->Scale(-opts_.denominator_scale);
  denominators_deriv_.Scale(-opts_.denominator_scale);

  // this function *adds to* its output.
  positive_computation_->Backward(nnet_output_deriv,
                                  &denominators_deriv_);

  // deriv w.r.t. exp of nnet output, as used in denominator computation.
  CuMatrix<BaseFloat> exp_nnet_output_deriv(nnet_output_.NumRows(),
                                            nnet_output_.NumCols());

  exp_nnet_output_deriv.AddMatMat(1.0, denominators_deriv_, kNoTrans,
                                  cu_weights_, kNoTrans, 0.0);

  // After the following statement, 'exp_nnet_output_deriv' contains the
  // derivative with respect to 'nnet_output_' itself, *considering only the
  // denominator term* of the positive computation.  We use that y/d(exp x) =
  // exp(x) dy/dx.
  exp_nnet_output_deriv.MulElements(exp_nnet_output_);
  nnet_output_deriv->AddMat(1.0, exp_nnet_output_deriv);

  if (supervision_.weight != 1.0) {
    // should be rare.  scale the derivatives.
    nnet_output_deriv->Scale(supervision_.weight);
  }
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


/**
   Computation of
     p(data | all classes).
   It's a HMM with num-states == num-history-states.  Each state has the same
   number of out-transitions, equal to num-phones + 1.

   Divide by p(data | all classes).  Forward-backward computation, with alphas and
   betas.  Store as non-log, but


     p(data | all classes), which we have to divide by, is the product over all
     frames of (non-tombstone-prob-mass), so the objf contains a term

     - \sum_{all frames} \log  (1 - tombstone-prob-mass)

     - \sum_{all frames} \log  (1 - tombstone-prob-mass)

  =  - \sum_{all frames} \log ( (denominator-sum - tombstone-sum) / denominator-sum ).

  =   \sum_{all frames} \log(denominator-sum) - \log (denominator-sum - tombstone-sum).

  so dObjf/d(denominator-sum) on frame t is
      1/(denominator-sum on frame t) - 1/(denominator-sum - tombstone-sum on frame t).

   = 1/d * (1 - 1/(1 - tombstone-prob)).

  and d/d(tombstone-sum) on frame t is:
     1/(denominator-sum - tombstone-sum).
  so dObjf/d(log tombstone-sum) on frame t is:
     tombstone-sum/(denominator-sum - tombstone-sum).

  Computation for a sequence of frames, t == 0 ... T - 1.
  Have two vectors 'occupancy0' and 'occupancy1', for
  even and odd times, of dimension == num-history-states.

  Initialize occupancy[*] = 1/num-history-states [for time t=0].
  # note: we go to only t - 2 because there is no point dealing
  # with end effects.  We should probably get rid of derivatives
  # near the edges anyway, they won't be accurate.

  # note: we imagine we are in a CUDA kernel and we have a range
  # of history-states that we are responsible for, and a particular
  # minibatch-element that we are responsible for.

  # the grid is one-dimensional and splits over the element of the
  # minibatch.
  # the block is one-dimensional and splits over the range of
  # history-states that this thread is responsible for.

  for t = 0 ... t - 2:
    if t % 2 == 0,
       this_occupancy = occupancy0;
       next_occupancy = occupancy1;
    else
       this_occupancy = occupancy0;
       next_occupancy = occupancy1;
    fi
    for j=0:num_history_states-1:
      # in CUDA, only handle a range of history-states in this j.
      next_occupancy[j] = 0.0
    done
    # note: we assume at this point that 'this_occupancy' is normalized
    # to sum to one.
    for j=0:num_history_states-1:
      occ = this_occupancy[j].
      if (occ very tiny) continue;
      this_factor = this_occ / this_den
      for k=0:num_history_states-1:  # in CUDA, only handle a range of history-states in this k.
         next_occupancy[k] += this_num * this_factor.
      done
    done
    #



  done


 */


}  // namespace ctc
}  // namespace kaldi

// chain/chain-numerator.cc

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


#include "chain/chain-numerator.h"
#include "cudamatrix/cu-vector.h"

namespace kaldi {
namespace chain {


NumeratorComputation::NumeratorComputation(
    const Supervision &supervision,
    const CuMatrixBase<BaseFloat> &nnet_output):
    supervision_(supervision),
    nnet_output_(nnet_output) {
  KALDI_ASSERT(supervision.num_sequences * supervision.frames_per_sequence ==
               nnet_output.NumRows() &&
               supervision.label_dim == nnet_output.NumCols());
}


void NumeratorComputation::ComputeLookupIndexes(
    std::vector<std::vector<int32> > *initial_pdf_ids,
    std::vector<std::vector<int32> > *final_pdf_ids) {
  std::vector<int32> fst_state_times;
  ComputeFstStateTimes(supervision_.fst, &fst_state_times);

  if (initial_pdf_ids)
    ComputeAllowedInitialAndFinalSymbols(supervision_,
                                         fst_state_times,
                                         initial_pdf_ids,
                                         final_pdf_ids);


  int32 num_states = supervision_.fst.NumStates();
  int32 num_arcs_guess = num_states * 2;
  fst_output_indexes_.reserve(num_arcs_guess);


  int32 frames_per_sequence = supervision_.frames_per_sequence,
      num_sequences = supervision_.num_sequences,
      cur_time = 0;

  // the following is a CPU version of nnet_output_indexes_.  It is a list of
  // pairs (row-index, column-index) which index nnet_output_.  The row-index
  // corresponds to the time-frame 't', and the column-index the pdf-id, but we
  // have to be a little careful with the row-index because there is a
  // reordering that happens if supervision_.num_sequences > 1.
  //

  // output-index) and denominator_indexes_cpu is a list of pairs (c,
  // history-state-index).
  std::vector<Int32Pair> nnet_output_indexes_cpu;

  // index_map_this_frame is a map, only valid for t == cur_time,
  // from the pdf-id to the index into nnet_output_indexes_cpu for the
  // likelihood at (cur_time, pdf-id).
  unordered_map<int32,int32> index_map_this_frame;

  typedef unordered_map<int32,int32>::iterator IterType;

  for (int32 state = 0; state < num_states; state++) {
    int32 t = fst_state_times[state];
    if (t != cur_time) {
      KALDI_ASSERT(t == cur_time + 1);
      index_map_this_frame.clear();
      cur_time = t;
    }
    for (fst::ArcIterator<fst::StdVectorFst> aiter(supervision_.fst, state);
         !aiter.Done(); aiter.Next()) {
      int32 pdf_id = aiter.Value().ilabel - 1;
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < supervision_.label_dim);

      int32 index = nnet_output_indexes_cpu.size();

      // the next few lines are a more efficient way of doing the following:
      // if (index_map_this_frame.count(pdf_id) == 0) {
      //   index = index_map_this_frame[pdf_id] = nnet_output_indexes_cpu.size();
      //   nnet_output_indexes_cpu.push_back(pair(pdf_id, row-index));
      // } else {
      //   index = index_map_this_frame[pdf_id];
      // }
      std::pair<IterType,bool> p = index_map_this_frame.insert(
          std::pair<const int32, int32>(pdf_id, index));
      if (p.second) {  // Was inserted -> map had no key 'output_index'
        Int32Pair pair;  // we can't use constructors as this was declared in C.
        pair.first = ComputeRowIndex(t, frames_per_sequence, num_sequences);
        pair.second = pdf_id;
        nnet_output_indexes_cpu.push_back(pair);
      } else {  // was not inserted -> set 'index' to the existing index.
        index = p.first->second;
      }
      fst_output_indexes_.push_back(index);
    }
  }
  nnet_output_indexes_ = nnet_output_indexes_cpu;
  KALDI_ASSERT(!fst_output_indexes_.empty());
}

void NumeratorComputation::GetAllowedInitialAndFinalSymbols(
    std::vector<std::vector<int32> > *initial_pdf_ids,
    std::vector<std::vector<int32> > *final_pdf_ids) {
  KALDI_ASSERT(fst_output_indexes_.empty());  // or functions called in wrong
                                              // order.
  ComputeLookupIndexes(initial_pdf_ids, final_pdf_ids);
}


BaseFloat NumeratorComputation::Forward() {
  if (fst_output_indexes_.empty()) {
    // if we haven't called this via GetAllowedInitialAndFinalSymbols()
    ComputeLookupIndexes(NULL, NULL);
  }
  nnet_logprobs_.Resize(nnet_output_indexes_.Dim(), kUndefined);
  nnet_output_.Lookup(nnet_output_indexes_, nnet_logprobs_.Data());
  const fst::StdVectorFst &fst = supervision_.fst;
  KALDI_ASSERT(fst.Start() == 0);
  int32 num_states = fst.NumStates();
  log_alpha_.Resize(num_states, kUndefined);
  log_alpha_.Set(-std::numeric_limits<double>::infinity());
  tot_log_prob_ = -std::numeric_limits<double>::infinity();

  log_alpha_(0) = 0.0;  // note, state zero is the start state, we checked above

  const BaseFloat *nnet_logprob_data = nnet_logprobs_.Data();
  std::vector<int32>::const_iterator fst_output_indexes_iter =
      fst_output_indexes_.begin();

  double *log_alpha_data = log_alpha_.Data();

  for (int32 state = 0; state < num_states; state++) {
    double this_log_alpha = log_alpha_data[state];
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state); !aiter.Done();
         aiter.Next(), ++fst_output_indexes_iter) {
      int32 nextstate = aiter.Value().nextstate;
      int32 index = *fst_output_indexes_iter;
      BaseFloat pseudo_loglike = nnet_logprob_data[index];
      double &next_log_alpha = log_alpha_data[nextstate];
      next_log_alpha = LogAdd(next_log_alpha, pseudo_loglike + this_log_alpha);
    }
    if (fst.Final(state) != fst::TropicalWeight::Zero())
      tot_log_prob_ = LogAdd(tot_log_prob_, this_log_alpha);
  }
  KALDI_ASSERT(fst_output_indexes_iter ==
               fst_output_indexes_.end());
  return tot_log_prob_;
}


void NumeratorComputation::Backward(
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  const fst::StdVectorFst &fst = supervision_.fst;
  int32 num_states = fst.NumStates();
  log_beta_.Resize(num_states, kUndefined);
  nnet_logprob_derivs_.Resize(nnet_logprobs_.Dim());

  // we'll be counting backwards and moving the 'fst_output_indexes_iter'
  // pointer back.
  const int32 *fst_output_indexes_iter = &(fst_output_indexes_[0]) +
      fst_output_indexes_.size();
  const BaseFloat *nnet_logprob_data = nnet_logprobs_.Data();
  double tot_log_prob = tot_log_prob_;
  double *log_beta_data = log_beta_.Data();
  const double *log_alpha_data = log_alpha_.Data();
  BaseFloat *nnet_logprob_deriv_data = nnet_logprob_derivs_.Data();

  for (int32 state = num_states - 1; state >= 0; state--) {
    int32 this_num_arcs  = fst.NumArcs(state);
    // on the backward pass we access the fst_output_indexes_ vector in a zigzag
    // pattern.
    fst_output_indexes_iter -= this_num_arcs;
    const int32 *this_fst_output_indexes_iter = fst_output_indexes_iter;
    double this_log_beta = (fst.Final(state) == fst::TropicalWeight::Zero() ?
                            -std::numeric_limits<double>::infinity() : 0.0);
    double this_log_alpha = log_alpha_data[state];
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state); !aiter.Done();
         aiter.Next(), this_fst_output_indexes_iter++) {
      double next_log_beta = log_beta_data[aiter.Value().nextstate];
      int32 index = *this_fst_output_indexes_iter;
      BaseFloat pseudo_loglike = nnet_logprob_data[index];
      this_log_beta = LogAdd(this_log_beta, pseudo_loglike + next_log_beta);
      BaseFloat occupation_logprob = this_log_alpha + pseudo_loglike +
          next_log_beta - tot_log_prob,
          occupation_prob = exp(occupation_logprob);
      nnet_logprob_deriv_data[index] += occupation_prob;
    }
    // check for -inf.
    KALDI_PARANOID_ASSERT(this_log_beta - this_log_beta == 0);
    log_beta_data[state] = this_log_beta;
  }
  KALDI_ASSERT(fst_output_indexes_iter == &(fst_output_indexes_[0]));

  int32 start_state = 0;  // the fact that the start state is numbered 0 is
                          // implied by other properties of the FST
                          // (epsilon-free-ness and topological sorting, and
                          // connectedness).
  double tot_log_prob_backward = log_beta_(start_state);
  if (!ApproxEqual(tot_log_prob_backward, tot_log_prob_))
    KALDI_WARN << "Disagreement in forward/backward log-probs: "
               << tot_log_prob_backward << " vs. " << tot_log_prob_;

  // copy this data to GPU.
  CuVector<BaseFloat> nnet_logprob_deriv_cuda;
  nnet_logprob_deriv_cuda.Swap(&nnet_logprob_derivs_);
  nnet_output_deriv->AddElements(1.0, nnet_output_indexes_,
                                 nnet_logprob_deriv_cuda.Data());
}


void ComputeAllowedInitialAndFinalSymbols(
    const Supervision &supervision,
    const std::vector<int32> &fst_state_times,
    std::vector<std::vector<int32> > *initial_symbols,
    std::vector<std::vector<int32> > *final_symbols) {
  int32 frames_per_sequence = supervision.frames_per_sequence,
      num_sequences = supervision.num_sequences,
      num_states = fst_state_times.size();
  initial_symbols->clear();
  initial_symbols->resize(num_sequences);
  final_symbols->clear();
  final_symbols->resize(num_sequences);
  for (int32 s = 0; s < num_states; s++) {
    int32 t = fst_state_times[s],
        sequence = t / frames_per_sequence,
        frame = t % frames_per_sequence;
    std::vector<int32> *target;
    if (frame == 0) {
      target = &((*initial_symbols)[sequence]);
    } else if (frame == frames_per_sequence - 1) {
      target = &((*final_symbols)[sequence]);
    } else {
      continue;
    }
    for (fst::ArcIterator<fst::StdVectorFst> aiter(supervision.fst, s);
         !aiter.Done(); aiter.Next()) {
      int32 pdf_id = aiter.Value().ilabel - 1;
      target->push_back(pdf_id);
    }
  }
  for (int32 seq = 0; seq < num_sequences; seq++) {
    KALDI_ASSERT(!(*initial_symbols)[seq].empty() &&
                 !(*final_symbols)[seq].empty());
    SortAndUniq(&((*initial_symbols)[seq]));
    SortAndUniq(&((*final_symbols)[seq]));
  }
}



}  // namespace chain
}  // namespace kaldi

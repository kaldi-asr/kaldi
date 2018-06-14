// rnnlm/rnnlm-lattice-rescoring.cc

// Copyright 2017 Johns Hopkins University (author: Daniel Povey) 
//           2017 Yiming Wang
//           2017 Hainan Xu
//
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

#include <utility>

#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {
namespace rnnlm {

KaldiRnnlmDeterministicFst::~KaldiRnnlmDeterministicFst() {
  int32 size = state_to_rnnlm_state_.size();
  for (int32 i = 0; i < size; i++)
    delete state_to_rnnlm_state_[i];
  
  state_to_rnnlm_state_.resize(0);
  state_to_wseq_.resize(0);
  wseq_to_state_.clear();
}

void KaldiRnnlmDeterministicFst::Clear() {
  // This function is similar to the destructor but we retain the 0-th entries
  // in each map which corresponds to the <bos> state.
  int32 size = state_to_rnnlm_state_.size();
  for (int32 i = 1; i < size; i++)
    delete state_to_rnnlm_state_[i];
  
  state_to_rnnlm_state_.resize(1);
  state_to_wseq_.resize(1);
  wseq_to_state_.clear();
  wseq_to_state_[state_to_wseq_[0]] = 0;
}

KaldiRnnlmDeterministicFst::KaldiRnnlmDeterministicFst(int32 max_ngram_order,
    const RnnlmComputeStateInfo &info) {
  max_ngram_order_ = max_ngram_order;
  bos_index_ = info.opts.bos_index;
  eos_index_ = info.opts.eos_index;

  std::vector<Label> bos_seq;
  bos_seq.push_back(bos_index_);
  state_to_wseq_.push_back(bos_seq);
  RnnlmComputeState *decodable_rnnlm = new RnnlmComputeState(info, bos_index_);
  wseq_to_state_[bos_seq] = 0;
  start_state_ = 0;

  state_to_rnnlm_state_.push_back(decodable_rnnlm);
}

fst::StdArc::Weight KaldiRnnlmDeterministicFst::Final(StateId s) {
  /// At this point, we have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  RnnlmComputeState* rnn = state_to_rnnlm_state_[s];
  return Weight(-rnn->LogProbOfWord(eos_index_));
}

bool KaldiRnnlmDeterministicFst::GetArc(StateId s, Label ilabel,
                                        fst::StdArc *oarc) {
  /// At this point, we have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> word_seq = state_to_wseq_[s];
  const RnnlmComputeState* rnnlm = state_to_rnnlm_state_[s];

  BaseFloat logprob = rnnlm->LogProbOfWord(ilabel);

  word_seq.push_back(ilabel);
  if (max_ngram_order_ > 0) {
    while (word_seq.size() >= max_ngram_order_) {
      /// History state has at most <max_ngram_order_> - 1 words in the state.
      word_seq.erase(word_seq.begin(), word_seq.begin() + 1);
    }
  }

  std::pair<const std::vector<Label>, StateId> wseq_state_pair(
      word_seq, static_cast<Label>(state_to_wseq_.size()));

  // Attemps to insert the current <wseq_state_pair>. If the pair already exists
  // then it returns false.
  typedef MapType::iterator IterType;
  std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

  // If the pair was just inserted, then also add it to state_to_* structures.
  if (result.second == true) {
    RnnlmComputeState *rnnlm2 = rnnlm->GetSuccessorState(ilabel);
    state_to_wseq_.push_back(word_seq);
    state_to_rnnlm_state_.push_back(rnnlm2);
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);
  return true;
}

}  // namespace rnnlm
}  // namespace kaldi

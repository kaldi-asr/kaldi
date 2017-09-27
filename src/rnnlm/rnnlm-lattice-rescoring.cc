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
namespace nnet3 {

KaldiRnnlmDeterministicFst::~KaldiRnnlmDeterministicFst() {
  int size = state_to_rnnlm_state_.size();
  KALDI_ASSERT(state_to_nnet3_output_.size() == size);
  for (int i = 0; i < size; i++) {
    delete state_to_rnnlm_state_[i];
    delete state_to_nnet3_output_[i];
  }
  
  state_to_rnnlm_state_.resize(0);
  state_to_nnet3_output_.resize(0);
  state_to_wseq_.resize(0);
  wseq_to_state_.clear();
}

void KaldiRnnlmDeterministicFst::Clear() {
  int size = state_to_rnnlm_state_.size();
  KALDI_ASSERT(state_to_nnet3_output_.size() == size);
  for (int i = 1; i < size; i++) {
    delete state_to_rnnlm_state_[i];
    delete state_to_nnet3_output_[i];
  }
  
  state_to_rnnlm_state_.resize(1);
  state_to_nnet3_output_.resize(1);
  state_to_wseq_.resize(1);
  wseq_to_state_.clear();
  wseq_to_state_[state_to_wseq_[0]] = 0;
}

void KaldiRnnlmDeterministicFst::ReadFstWordSymbolTableAndRnnWordlist(
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename) {
  // Reads symbol table.
  fst::SymbolTable *fst_word_symbols = NULL;
  if (!(fst_word_symbols =
      fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
              << word_symbol_table_rxfilename;
  }

  full_voc_size_ = fst_word_symbols->NumSymbols();
  fst_label_to_word_.resize(full_voc_size_);

  for (int32 i = 0; i < fst_label_to_word_.size(); ++i) {
    fst_label_to_word_[i] = fst_word_symbols->Find(i);
    if (fst_label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
                << "symbol table, mismatched symbol table or you have discoutinuous "
                << "integers in your symbol table?";
    }
  }

  fst_label_to_rnn_label_.resize(fst_word_symbols->NumSymbols(), -1);

  oos_index_ = -2;  // use -2 since fst::SymbolTable::kNoSymbol is -1
  {
    std::ifstream ifile(rnn_wordlist.c_str());
    int32 id;
    string word;
    int32 i = 0;
    while (ifile >> word >> id) {
      if (word == eos_symbol_) {
        eos_index_ = id;
      } else if (word == bos_symbol_) {
        bos_index_ = id;
      } else if (word == oos_symbol_) {
        oos_index_ = id;
      } else if (word == brk_symbol_) {
        brk_index_ = id;
      }
      KALDI_ASSERT(i == id);
      i++;
      rnn_label_to_word_.push_back(word);

      int fst_label = fst_word_symbols->Find(rnn_label_to_word_[id]);
      if (fst::SymbolTable::kNoSymbol != fst_label && id != oos_index_
               && id != bos_index_ && id != brk_index_) {
        KALDI_LOG << "warning: word " << word
                  << " in RNNLM wordlist but not in FST wordlist";
      }
      if (id != oos_index_ && oos_index_ != -2 &&
                fst_label != fst::SymbolTable::kNoSymbol) {
        fst_label_to_rnn_label_[fst_label] = id;
      }
    }
  }

  for (int32 i = 0; i < fst_label_to_rnn_label_.size(); i++) {
    if (fst_label_to_rnn_label_[i] == -1) {
      fst_label_to_rnn_label_[i] = oos_index_;
    }
  }
  delete fst_word_symbols;
}

KaldiRnnlmDeterministicFst::KaldiRnnlmDeterministicFst(int32 max_ngram_order,
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename,
    const RnnlmComputeStateInfo &info) {
  max_ngram_order_ = max_ngram_order;
  bos_symbol_ = info.opts.bos_symbol;
  eos_symbol_ = info.opts.eos_symbol;
  oos_symbol_ = info.opts.oos_symbol;
  brk_symbol_ = info.opts.brk_symbol;
  ReadFstWordSymbolTableAndRnnWordlist(rnn_wordlist,
                                       word_symbol_table_rxfilename);

  std::vector<Label> bos_seq;
  bos_seq.push_back(bos_index_);
  state_to_wseq_.push_back(bos_seq);
  RnnlmComputeState *decodable_rnnlm = new RnnlmComputeState(info);
  decodable_rnnlm->TakeFeatures(bos_index_);
  CuVector<BaseFloat> *hidden = decodable_rnnlm->GetOutput();
  wseq_to_state_[bos_seq] = 0;
  start_state_ = 0;

  state_to_rnnlm_state_.push_back(decodable_rnnlm);
  state_to_nnet3_output_.push_back(hidden);
}

fst::StdArc::Weight KaldiRnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  const CuVector<BaseFloat> &nnet3_out = *state_to_nnet3_output_[s];
  RnnlmComputeState* rnn = state_to_rnnlm_state_[s];
  return rnn->LogProbOfWord(eos_index_, nnet3_out);
}

bool KaldiRnnlmDeterministicFst::GetArc(StateId s, Label ilabel,
                                        fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  const RnnlmComputeState* rnnlm = state_to_rnnlm_state_[s];
  int32 rnn_word = fst_label_to_rnn_label_[ilabel];

  const CuVector<BaseFloat> &nnet3_out = *state_to_nnet3_output_[s];
  BaseFloat logprob = rnnlm->LogProbOfWord(rnn_word, nnet3_out);

//  if (rnn_word == oos_index_)
//    logprob = logprob - Log(full_voc_size_ - rnn_label_to_word_.size() + 1.0);

  wseq.push_back(rnn_word);
  if (max_ngram_order_ > 0) {
    while (wseq.size() >= max_ngram_order_) {
      // History state has at most <max_ngram_order_> - 1 words in the state.
      wseq.erase(wseq.begin(), wseq.begin() + 1);
    }
  }

  std::pair<const std::vector<Label>, StateId> wseq_state_pair(
      wseq, static_cast<Label>(state_to_wseq_.size()));

  // Attemps to insert the current <lseq_state_pair>. If the pair already exists
  // then it returns false.
  typedef MapType::iterator IterType;
  std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

  // If the pair was just inserted, then also add it to state_to_* structures
  if (result.second == true) {
    RnnlmComputeState *rnnlm2 = new RnnlmComputeState(*rnnlm);  // make a copy
    rnnlm2->TakeFeatures(rnn_word);
    state_to_wseq_.push_back(wseq);
    state_to_nnet3_output_.push_back(rnnlm2->GetOutput());
    state_to_rnnlm_state_.push_back(rnnlm2);
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

}  // namespace nnet3
}  // namespace kaldi

// lm/kaldi-rnnlm.cc

// Copyright 2015  Guoguo Chen

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

#include "lm/kaldi-cuedrnnlm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

KaldiCuedRnnlmWrapper::KaldiCuedRnnlmWrapper(
    const KaldiCuedRnnlmWrapperOpts &opts,
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename,
    const std::string &rnnlm_rxfilename):
       rnnlm_(rnnlm_rxfilename, rnn_wordlist, rnn_wordlist, opts.LayerSizes(),
              opts.full_voc_size, false, 0) {

  // Reads symbol table.
  fst::SymbolTable *fst_word_symbols = NULL;
  if (!(fst_word_symbols =
        fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
        << word_symbol_table_rxfilename;
  }

  fst_label_to_word_.resize(fst_word_symbols->NumSymbols());

  for (int32 i = 0; i < fst_label_to_word_.size(); ++i) {
    fst_label_to_word_[i] = fst_word_symbols->Find(i);
    if (fst_label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }

//  fst::SymbolTable *rnn_word_symbols = NULL;
/*
  if (!(rnn_word_symbols =
        fst::SymbolTable::ReadText(rnn_wordlist))) {
    KALDI_ERR << "Could not read symbol table from file "
        << rnn_wordlist;
  }
*/

//  rnn_label_to_word_.resize(rnn_word_symbols->NumSymbols() + 2);
  fst_label_to_rnn_label_.resize(fst_word_symbols->NumSymbols(), -1);
                                 // +1 is the <OOS> symbol
//                                 rnn_word_symbols->NumSymbols() + 1);

  rnn_label_to_word_.push_back("<s>");
  { // input.
    ifstream ifile(rnn_wordlist.c_str());
    int id;
    string word;
    int i = 0;
    while (ifile >> id >> word) {
      if (word == "[UNK]") {
        word = "<unk>";
      } else if (word == "<OOS>") {
        continue;
      }
      i++;
      KALDI_ASSERT(i == id + 1);
      rnn_label_to_word_.push_back(word);

      int fst_label = fst_word_symbols->Find(rnn_label_to_word_[i]);
      KALDI_ASSERT(fst::SymbolTable::kNoSymbol != fst_label);
      fst_label_to_rnn_label_[fst_label] = i;
    }
  }
  rnn_label_to_word_.push_back("<OOS>");
  
  for (int i = 0; i < fst_label_to_rnn_label_.size(); i++) {
    if (fst_label_to_rnn_label_[i] == -1) {
      fst_label_to_rnn_label_[i] = rnn_label_to_word_.size() - 1;
    }
  }
}

BaseFloat KaldiCuedRnnlmWrapper::GetLogProb(
    int32 word, const std::vector<int32> &wseq,
    const std::vector<BaseFloat> &context_in,
    std::vector<BaseFloat> *context_out) {

  BaseFloat logprob = rnnlm_.computeConditionalLogprob(word, wseq,
                                                       context_in, context_out);
  return logprob;
}

CuedRnnlmDeterministicFst::CuedRnnlmDeterministicFst(int32 max_ngram_order,
                                             KaldiCuedRnnlmWrapper *rnnlm) {
  KALDI_ASSERT(rnnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  rnnlm_ = rnnlm;
  rnnlm_->ResetHistory();

  std::vector<Label> bos;
  bos.push_back(0); // 0 for <s>
  std::vector<BaseFloat> bos_context(rnnlm->GetHiddenLayerSize(), 0.1f);
  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(bos_context);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight CuedRnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  BaseFloat logprob = rnnlm_->GetLogProb(0, wseq, state_to_context_[s], NULL);
  return Weight(-logprob);
}

bool CuedRnnlmDeterministicFst::GetArc(StateId s, Label ilabel, fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  std::vector<BaseFloat> new_context(rnnlm_->GetHiddenLayerSize());

  int32 rnn_word = rnnlm_->fst_label_to_rnn_label_[ilabel];
  BaseFloat logprob = rnnlm_->GetLogProb(rnn_word, wseq,
                                         state_to_context_[s], &new_context);

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

  // If the pair was just inserted, then also add it to <state_to_wseq_> and
  // <state_to_context_>.
  if (result.second == true) {
    state_to_wseq_.push_back(wseq);
    state_to_context_.push_back(new_context);
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

}  // namespace kaldi

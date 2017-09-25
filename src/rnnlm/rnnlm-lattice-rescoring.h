// rnnlm/rnnlm-lattice-rescoring.h
//
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

#ifndef KALDI_LM_KALDI_RNNLM_RESCORING_H_
#define KALDI_LM_KALDI_RNNLM_RESCORING_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "rnnlm/rnnlm-simple-looped.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet3 {

class KaldiRnnlmDeterministicFst
    : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  KaldiRnnlmDeterministicFst(int32 max_ngram_order,
      const std::string &rnn_wordlist,
      const std::string &word_symbol_table_rxfilename,
      const RnnlmSimpleLoopedInfo &info);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  std::vector<int32> fst_label_to_rnn_label_;
  std::vector<std::string> rnn_label_to_word_;

//  std::vector<int32> fst_label_to_rnn_in_label_;
//  std::vector<std::string> rnn_in_label_to_word_;

  std::vector<std::string> fst_label_to_word_;
  int32 full_voc_size_;
  int32 out_OOS_index_;
  int32 bos_index_;
  int32 eos_index_;

  typedef unordered_map
      <std::vector<Label>, StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;

  int32 max_ngram_order_;
  std::vector<RnnlmSimpleLooped> state_to_decodable_rnnlm_;

  void ReadFstWordSymbolTableAndRnnWordlist(const std::string &rnn_in_wordlist,
//      const std::string &rnn_out_wordlist,
      const std::string &word_symbol_table_rxfilename);

};

}  // namespace nnet3
}  // namespace kaldi

#endif  // KALDI_LM_KALDI_RNNLM_RESCORING_H_

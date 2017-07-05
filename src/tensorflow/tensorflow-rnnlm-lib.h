// tensorflow-rnnlm-lib.h

// Copyright         2017 Hainan Xu

// wrapper for tensorflow rnnlm

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

#ifndef KALDI_LM_TENSORFLOW_LIB_H_
#define KALDI_LM_TENSORFLOW_LIB_H_

#include <string>
#include <vector>
#include "util/stl-utils.h"
#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Session;
using tensorflow::Tensor;

namespace kaldi {
namespace tf_rnnlm {

struct KaldiTfRnnlmWrapperOpts {
  std::string unk_symbol;
  std::string eos_symbol;

  KaldiTfRnnlmWrapperOpts() : unk_symbol("<oos>"), eos_symbol("</s>") {}

  void Register(OptionsItf *opts) {
    opts->Register("unk-symbol", &unk_symbol, "Symbol for out-of-vocabulary "
                   "words in rnnlm.");
    opts->Register("eos-symbol", &eos_symbol, "End of setence symbol in "
                   "rnnlm.");
  }
};

class KaldiTfRnnlmWrapper {
 public:
  KaldiTfRnnlmWrapper(const KaldiTfRnnlmWrapperOpts &opts,
                      const std::string &rnn_wordlist,
                      const std::string &word_symbol_table_rxfilename,
                      const std::string &unk_prob_file,
                      const std::string &tf_model_path);

  ~KaldiTfRnnlmWrapper() {
    session_->Close();
  }

  int32 GetEos() const { return eos_; }

  // get an all-zero Tensor of the size that matches the hidden state of the TF model
  const Tensor& GetInitialContext() const;

  // get the 2nd-to-last layer of RNN when feeding input of
  // (initial-context, sentence-boundary)
  const Tensor& GetInitialCell() const;

  // compute p(word | wseq) and return the log of that
  // the computation used the input cell,
  // which is the 2nd-to-last layer of the RNNLM associated with history wseq;
  //
  // and we generate (context_out, new_cell) by passing (context_in, word) into the model
  // if the last 2 pointers are NULL we don't query then in TF session
  BaseFloat GetLogProb(int32 word, // need the FST word label for computing OOS cost
                       int32 fst_word,
                       const Tensor &context_in,  // context to pass into RNN
                       const Tensor &cell_in,     // 2nd-to-last layer
                       Tensor *context_out,
                       Tensor *new_cell);

  // since usually we have a smaller vocab in RNN than the whole vocab,
  // we use this mapping during rescoring
  std::vector<int> fst_label_to_rnn_label_;
  std::vector<std::string> rnn_label_to_word_;
  std::vector<std::string> fst_label_to_word_;
 private:
  void ReadTfModel(const std::string &tf_model_path);

  // do queries on the session to get the initial tensors (cell + context)
  void AcquireInitialTensors();

  KaldiTfRnnlmWrapperOpts opts_;
  Tensor initial_context_;
  Tensor initial_cell_;

  // this corresponds to the FST symbol table
  int32 num_total_words;
  // this corresponds to the RNNLM symbol table
  int32 num_rnn_words;

  Session* session_;  // for TF computation; pointer owned here
  int32 eos_;
  int32 oos_;

  std::vector<float> unk_costs_;  // extra cost for OOS symbol in RNNLM

  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiTfRnnlmWrapper);
};

class TfRnnlmDeterministicFst
    : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  TfRnnlmDeterministicFst(int32 max_ngram_order, KaldiTfRnnlmWrapper *rnnlm);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;

  KaldiTfRnnlmWrapper *rnnlm_;
  int32 max_ngram_order_;
  std::vector<tensorflow::Tensor> state_to_context_;
  std::vector<tensorflow::Tensor> state_to_cell_; // cell is the 2nd-to-last output of RNN
};

}  // namespace tf_rnnlm
}  // namespace kaldi

#endif  // KALDI_LM_TENSORFLOW_LIB_H_

// tensorflow-rnnlm.h

// Copyright (C) 2017 Intellisist, Inc. (Author: Hainan Xu)

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

#ifndef KALDI_TFRNNLM_TENSORFLOW_RNNLM_H_
#define KALDI_TFRNNLM_TENSORFLOW_RNNLM_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "util/stl-utils.h"
#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"

// Following macros are defined in both OpenFst and Tensorflow headers. Here we
// undef them before including "tensorflow/core/public/session.h" to silence
// compiler warnings. Note that this is not a panacea. We should still pay
// attention to the order of includes in other places in the codebase to avoid
// using the wrong macro definitions. Any OpenFst header or any header including
// an OpenFst header should be included before tfrnnlm/tensorflow-rnnlm.h. Also,
// to avoid macro redefinitions, any Tensorflow header should be included after
// tfrnnlm/tensorflow-rnnlm.h.
#undef LOG
#undef VLOG
#undef CHECK
#undef CHECK_EQ
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_NE
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_LT
#undef DCHECK_GT
#undef DCHECK_LE
#undef DCHECK_GE
#undef DCHECK_NE

#include "tensorflow/cc/saved_model/loader.h"

using tensorflow::Session;
using tensorflow::Tensor;

namespace kaldi {
namespace tf_rnnlm {

struct KaldiTfRnnlmWrapperOpts {
  std::string unk_symbol;
  int32 num_threads;  // 0 means unlimited

  KaldiTfRnnlmWrapperOpts() : unk_symbol("<oos>"), num_threads(1) {}

  void Register(OptionsItf *opts) {
    opts->Register("unk-symbol", &unk_symbol, "Symbol for out-of-vocabulary "
                   "words in rnnlm.");
    opts->Register("num-threads", &num_threads, "Number of threads for TF computation; "
                   "0 means unlimited.");
  }
};

/**
This class wraps the TensorFlow based RNNLM, and provides a set of interfaces
to be used for class TfRnnlmDeterministicFst, implemented later in this file
*/
class KaldiTfRnnlmWrapper {
 public:

  /// constructor
  /// opts specify symbol for <unk> and num-threads for computation
  /// rnn_wordlist specifies a wordlist file with format
  ///        [int-word-id] [word]
  ///     the word <oos> must appear in this file
  //
  /// word_symbol_table_rxfilename points to a standard word-list file in OpenFST style
  /// unk_prob_file has the format
  ///        [word]  [prob or count]  (it auto-normalizes the probabilities)
  /// tf_model_path is the location of the TensorFlow model
  KaldiTfRnnlmWrapper(const KaldiTfRnnlmWrapperOpts &opts,
                      const std::string &rnn_wordlist,
                      const std::string &word_symbol_table_rxfilename,
                      const std::string &unk_prob_file,
                      const std::string &tf_model_path);
  ~KaldiTfRnnlmWrapper();

  int32 GetEos() const { return eos_; }

  /// get an all-zero Tensor of the size that matches the hidden state of the TF model
  const Tensor& GetInitialContext() const;

  /// get the 2nd-to-last layer of RNN when feeding input of
  /// (initial-context, sentence-boundary)
  /// "cell" is short for "(last)cell-output"; calling it "cell" here because in
  /// later functions we have function GetLogProb() where we need to pass in
  /// one "cell" as input and another as output; to avoid confusing we use a single
  /// word "cell" for that instead of things like cell_out_in and cell_out_out.
  const Tensor& GetInitialCell() const;

  /// compute p(word | wseq) and return the log of that
  /// the computation used the input cell,
  /// which is the 2nd-to-last layer of the RNNLM associated with history wseq;
  ///
  /// and we generate (context_out, new_cell) by passing (context_in, word)
  /// into the TensorFlow session that manages the RNNLM
  /// if the last 2 pointers are NULL we don't query them in TF session
  /// e.g. in the case of computing p(</s>|some history)
  BaseFloat GetLogProb(int32 word,      // word id in RNN wordlist
                       int32 fst_word,  // FST word label, only for computing OOS cost
                       const Tensor &context_in,
                       const Tensor &cell_in,
                       Tensor *context_out,
                       Tensor *cell_out);

  /// takes in a word-id for FST and return the word-id for RNNLM
  /// return the word-id for <oos> if not found
  int FstLabelToRnnLabel(int i) const;

 private:
  /// read the TensorFlow model and create the session for computation
  /// num-threads need to be specified in creating the session
  void ReadTfModel(const std::string &tf_model_path, int32 num_threads);

  /// do queries on the session to get the initial tensors (cell + context)
  void AcquireInitialTensors();

  /// since usually we have a smaller vocab in RNN than the whole vocab,
  /// we use this mapping during rescoring
  std::vector<int> fst_label_to_rnn_label_;
  std::vector<std::string> rnn_label_to_word_;
  std::vector<std::string> fst_label_to_word_;

  KaldiTfRnnlmWrapperOpts opts_;
  Tensor initial_context_;
  Tensor initial_cell_;

  // this corresponds to the FST symbol table
  int32 num_total_words;
  // this corresponds to the RNNLM symbol table
  int32 num_rnn_words;

  // for TF computation
  tensorflow::SavedModelBundle bundle_;
  std::string word_id_tensor_name_;
  std::string context_tensor_name_;
  std::string log_prob_tensor_name_;
  std::string rnn_out_tensor_name_;
  std::string rnn_states_tensor_name_;
  std::string initial_state_tensor_name_;
  int32 eos_;
  int32 oos_;

  std::vector<float> unk_costs_;  // extra cost for OOS symbol in RNNLM

  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiTfRnnlmWrapper);
};

class TfRnnlmDeterministicFst:
         public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  TfRnnlmDeterministicFst(int32 max_ngram_order, KaldiTfRnnlmWrapper *rnnlm);
  ~TfRnnlmDeterministicFst();
  void Clear();

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
  std::vector<Tensor*> state_to_context_;
  std::vector<Tensor*> state_to_cell_;
};

}  // namespace tf_rnnlm
}  // namespace kaldi

#endif  // KALDI_TFRNNLM_TENSORFLOW_RNNLM_H_

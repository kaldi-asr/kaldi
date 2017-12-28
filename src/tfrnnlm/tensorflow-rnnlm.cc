// tensorflow-rnnlm.cc

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


#include <utility>
#include <fstream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "tfrnnlm/tensorflow-rnnlm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {
using std::ifstream;
using tf_rnnlm::KaldiTfRnnlmWrapper;
using tf_rnnlm::TfRnnlmDeterministicFst;
using tensorflow::Status;

// read a unigram count file of the OOSs and generate extra OOS costs for words
void SetUnkPenalties(const string &filename,
                     const fst::SymbolTable& fst_word_symbols,
                     std::vector<float> *out) {
  if (filename == "")
    return;
  out->resize(fst_word_symbols.NumSymbols(), 0);  // default is 0
  ifstream ifile(filename.c_str());
  string word;
  float count, total_count = 0;
  while (ifile >> word >> count) {
    int id = fst_word_symbols.Find(word);
    KALDI_ASSERT(id != fst::SymbolTable::kNoSymbol);
    (*out)[id] = count;
    total_count += count;
  }

  for (int i = 0; i < out->size(); i++) {
    if ((*out)[i] != 0) {
      (*out)[i] = log ((*out)[i] / total_count);
    }
  }
}

// Read tensorflow checkpoint files
void KaldiTfRnnlmWrapper::ReadTfModel(const std::string &tf_model_path,
                                      int32 num_threads) {
  string graph_path = tf_model_path + ".meta";

  tensorflow::SessionOptions session_options;

  session_options.config.set_intra_op_parallelism_threads(num_threads);
  session_options.config.set_inter_op_parallelism_threads(num_threads);

  Status status = tensorflow::NewSession(session_options,
                                         &session_);
  if (!status.ok()) {
    KALDI_ERR << status.ToString();
  }

  tensorflow::MetaGraphDef graph_def;
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path,
                                       &graph_def);
  if (!status.ok()) {
    KALDI_ERR << status.ToString();
  }

  // Add the graph to the session
  status = session_->Create(graph_def.graph_def());
  if (!status.ok()) {
    KALDI_ERR << status.ToString();
  }

  Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<std::string>()() = tf_model_path;

  status = session_->Run(
      {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor} },
      {},
      {graph_def.saver_def().restore_op_name()},
      nullptr);
  if (!status.ok()) {
    KALDI_ERR << status.ToString();
  }
}

KaldiTfRnnlmWrapper::KaldiTfRnnlmWrapper(
    const KaldiTfRnnlmWrapperOpts &opts,
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename,
    const std::string &unk_prob_file,
    const std::string &tf_model_path): opts_(opts) {
  ReadTfModel(tf_model_path, opts.num_threads);

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
      KALDI_ERR << "Could not find word for integer " << i << " in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }

  // first put all -1's; will check later
  fst_label_to_rnn_label_.resize(fst_word_symbols->NumSymbols(), -1);
  num_total_words = fst_word_symbols->NumSymbols();

  // read rnn wordlist and then generate ngram-label-to-rnn-label map
  oos_ = -1;
  { // input.
    ifstream ifile(rnn_wordlist.c_str());
    string word;
    int id = -1;
    eos_ = 0;
    while (ifile >> word) {
      id++;
      rnn_label_to_word_.push_back(word);  // vector[i] = word

      int fst_label = fst_word_symbols->Find(word);
      if (fst::SymbolTable::kNoSymbol == fst_label) {
        if (id == eos_)
          continue;

        KALDI_ASSERT(word == opts_.unk_symbol && oos_ == -1);
        oos_ = id;
        continue;
      }
      KALDI_ASSERT(fst_label >= 0);
      fst_label_to_rnn_label_[fst_label] = id;
    }
  }
  if (fst_label_to_word_.size() > rnn_label_to_word_.size()) {
    KALDI_ASSERT(oos_ != -1);
  }
  num_rnn_words = rnn_label_to_word_.size();

  // we must have an oos symbol in the wordlist
  if (oos_ == -1)
    return;

  for (int i = 0; i < fst_label_to_rnn_label_.size(); i++) {
    if (fst_label_to_rnn_label_[i] == -1) {
      fst_label_to_rnn_label_[i] = oos_;
    }
  }

  AcquireInitialTensors();
  SetUnkPenalties(unk_prob_file, *fst_word_symbols, &unk_costs_);
  delete fst_word_symbols;
}

void KaldiTfRnnlmWrapper::AcquireInitialTensors() {
  Status status;
  // get the initial context; this is basically the all-0 tensor
  {
    std::vector<Tensor> state;
    status = session_->Run(std::vector<std::pair<string, Tensor> >(),
                           {"Train/Model/test_initial_state"}, {}, &state);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }
    initial_context_ = state[0];
  }

  // get the initial pre-final-affine layer
  {
    std::vector<Tensor> state;
    Tensor bosword(tensorflow::DT_INT32, {1, 1});
    bosword.scalar<int32>()() = eos_;  // eos_ is more like a sentence boundary

    std::vector<std::pair<string, Tensor> > inputs = {
      {"Train/Model/test_word_in", bosword},
      {"Train/Model/test_state_in", initial_context_},
    };

    status = session_->Run(inputs, {"Train/Model/test_cell_out"}, {}, &state);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }
    initial_cell_ = state[0];
  }
}

BaseFloat KaldiTfRnnlmWrapper::GetLogProb(int32 word,
                                          int32 fst_word,
                                          const Tensor &context_in,
                                          const Tensor &cell_in,
                                          Tensor *context_out,
                                          Tensor *new_cell) {
  std::vector<std::pair<string, Tensor> > inputs;

  Tensor thisword(tensorflow::DT_INT32, {1, 1});

  thisword.scalar<int32>()() = word;
  std::vector<Tensor> outputs;

  if (context_out != NULL) {
    inputs = {
      {"Train/Model/test_word_in", thisword},
      {"Train/Model/test_word_out", thisword},
      {"Train/Model/test_state_in", context_in},
      {"Train/Model/test_cell_in", cell_in},
    };

    // The session will initialize the outputs
    // Run the session, evaluating our "c" operation from the graph
    Status status = session_->Run(inputs,
        {"Train/Model/test_out",
         "Train/Model/test_state_out",
         "Train/Model/test_cell_out"}, {}, &outputs);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }

    *context_out = outputs[1];
    *new_cell = outputs[2];
  } else {
    inputs = {
      {"Train/Model/test_word_out", thisword},
      {"Train/Model/test_cell_in", cell_in},
    };

    // Run the session, evaluating our "c" operation from the graph
    Status status = session_->Run(inputs,
        {"Train/Model/test_out"}, {}, &outputs);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }
  }

  float ans;
  if (word != oos_) {
    ans = outputs[0].scalar<float>()();
  } else {
    if (unk_costs_.size() == 0) {
      ans = outputs[0].scalar<float>()() - log(num_total_words - num_rnn_words);
    } else {
      ans = outputs[0].scalar<float>()() + unk_costs_[fst_word];
    }
  }

  return ans;
}

const Tensor& KaldiTfRnnlmWrapper::GetInitialContext() const {
  return initial_context_;
}

const Tensor& KaldiTfRnnlmWrapper::GetInitialCell() const {
  return initial_cell_;
}

int KaldiTfRnnlmWrapper::FstLabelToRnnLabel(int i) const {
  KALDI_ASSERT(i >= 0 && i < fst_label_to_rnn_label_.size());
  return fst_label_to_rnn_label_[i];
}

TfRnnlmDeterministicFst::TfRnnlmDeterministicFst(int32 max_ngram_order,
                                             KaldiTfRnnlmWrapper *rnnlm) {
  KALDI_ASSERT(rnnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  rnnlm_ = rnnlm;

  std::vector<Label> bos;
  const Tensor& initial_context = rnnlm_->GetInitialContext();
  const Tensor& initial_cell = rnnlm_->GetInitialCell();

  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(new Tensor(initial_context));
  state_to_cell_.push_back(new Tensor(initial_cell));
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

TfRnnlmDeterministicFst::~TfRnnlmDeterministicFst() {
  for (int i = 0; i < state_to_context_.size(); i++) {
    delete state_to_context_[i];
  }
  for (int i = 0; i < state_to_cell_.size(); i++) {
    delete state_to_cell_[i];
  }
}

void TfRnnlmDeterministicFst::Clear() {
  // similar to the destructor but we retain the 0-th entries in each map
  // which corresponds to the <bos> state
  for (int i = 1; i < state_to_context_.size(); i++) {
    delete state_to_context_[i];
  }
  for (int i = 1; i < state_to_cell_.size(); i++) {
    delete state_to_cell_[i];
  }
  
  state_to_context_.resize(1);
  state_to_cell_.resize(1);
  state_to_wseq_.resize(1);
  wseq_to_state_.clear();
  wseq_to_state_[state_to_wseq_[0]] = 0;
}


fst::StdArc::Weight TfRnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  BaseFloat logprob = rnnlm_->GetLogProb(rnnlm_->GetEos(),
                         -1,  // only need type; this param will not be used
                         *state_to_context_[s],
                         *state_to_cell_[s], NULL, NULL);
  return Weight(-logprob);
}

bool TfRnnlmDeterministicFst::GetArc(StateId s, Label ilabel,
                                     fst::StdArc *oarc) {
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  Tensor *new_context = new Tensor();
  Tensor *new_cell = new Tensor();

  // look-up the rnn label from the FST label
  int32 rnn_word = rnnlm_->FstLabelToRnnLabel(ilabel);
  BaseFloat logprob = rnnlm_->GetLogProb(rnn_word,
                                         ilabel,
                                         *state_to_context_[s],
                                         *state_to_cell_[s],
                                         new_context,
                                         new_cell);

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
    state_to_cell_.push_back(new_cell);
  } else {
    delete new_context;
    delete new_cell;
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

}  // namespace kaldi

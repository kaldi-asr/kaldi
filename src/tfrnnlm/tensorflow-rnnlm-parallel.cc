// tensorflow-rnnlm-parallel.cc

// Copyright (C) 2017 Intellisist, Inc. (Author: Hainan Xu)
//               2017 Dongji Gao, Hainan Xu

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
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

#include "tfrnnlm/tensorflow-rnnlm-parallel.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {
using std::ifstream;
using tf_rnnlm::KaldiTfRnnlmWrapper;
using tf_rnnlm::TfRnnlmDeterministicFstParallel;
using tensorflow::Status;


void tf_rnnlm::ConcatTensor(const std::vector<tensorflow::Tensor*> &input_tensor_vector,
                 const tensorflow::Scope &scope,
                 const tensorflow::ClientSession &session,
                 Tensor *output_tensor) {
  if (input_tensor_vector.size() == 1) {
    *output_tensor = *input_tensor_vector[0];
    return;
  }

  // convert vector<Tensor> to InputList
  std::vector<tensorflow::Input> v;
  for (int i = 0; i < input_tensor_vector.size(); i++) {
    v.push_back(tensorflow::Input(*input_tensor_vector[i]));
  }
  tensorflow::InputList inputlist = tensorflow::InputList(v);
  auto stack_op = tensorflow::ops::Concat(scope, inputlist, -2);

  std::vector<Tensor> stack_output_vector(1);

  Status status;
  status = session.Run({stack_op}, &stack_output_vector);
  if (!status.ok()) {
    KALDI_ERR << status.ToString();
  } else  {
    KALDI_ASSERT(stack_output_vector.size() == 1);
    *output_tensor = stack_output_vector[0];
  }
}

void tf_rnnlm::SplitTensor(int size,
                   const Tensor &input_tensor,
                   const tensorflow::Scope &scope,
                   const tensorflow::ClientSession &session,
                   std::vector<Tensor> *output_tensor_vector) {
  if (size == 1) {
    output_tensor_vector->resize(1);
    (*output_tensor_vector)[0] = input_tensor;
    return;
  }

  tensorflow::Input input = tensorflow::Input(input_tensor);
  auto unstack_op = tensorflow::ops::Split(scope, -2, input_tensor, size).output; 
  
  Status status;
  status = session.Run({unstack_op}, output_tensor_vector);

  if (!status.ok()) {
    KALDI_ERR << status.ToString();
  }
}

TfRnnlmDeterministicFstParallel::TfRnnlmDeterministicFstParallel(int32 max_ngram_order,
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

TfRnnlmDeterministicFstParallel::~TfRnnlmDeterministicFstParallel() {
  for (int i = 0; i < state_to_context_.size(); i++) {
    delete state_to_context_[i];
  }
  for (int i = 0; i < state_to_cell_.size(); i++) {
    delete state_to_cell_[i];
  }
}


void TfRnnlmDeterministicFstParallel::FinalParallel(std::vector<StateId> s2_vector_final,
                                                    std::vector<Weight>* det_fst_final_vector) {
  int32 eos_ = rnnlm_->GetEos();
  int s2_size = s2_vector_final.size();
  std::vector<int32> rnn_word_vector(s2_size, eos_);
  std::vector<Label> ilabel_vector(s2_size, -1);
  std::vector<tensorflow::Tensor*> state_to_context_vector, state_to_cell_vector;
  std::vector<BaseFloat> logprob_vector;
  tensorflow::Scope root_final = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session_final(root_final);

  for (int i = 0; i < s2_size; ++i) {
    StateId s = s2_vector_final[i];
    state_to_context_vector.push_back(state_to_context_[s]);
    state_to_cell_vector.push_back(state_to_cell_[s]);  
  }

  std::vector<Tensor*> state_output_vector, cell_output_vector;
  Tensor state_to_context_tensor, state_to_cell_tensor;

  tf_rnnlm::ConcatTensor(state_to_context_vector, root_final, session_final,
              &state_to_context_tensor);
  tf_rnnlm::ConcatTensor(state_to_cell_vector, root_final, session_final,
              &state_to_cell_tensor);

  rnnlm_->GetLogProbParallel(rnn_word_vector,
                             ilabel_vector,
                             state_to_context_tensor,
                             state_to_cell_tensor,
                             NULL,
                             NULL,
                             &logprob_vector);

  
  for (int i = 0; i < s2_vector_final.size(); ++i) {
    det_fst_final_vector->push_back(Weight(-logprob_vector[i]));
  }
}

// Parallel version of <GetArc>.
void TfRnnlmDeterministicFstParallel::GetArcsParallel(std::vector<StateId> s2_vector, 
                                              std::vector<Label> ilabel_vector,
                                              std::vector<fst::StdArc>* arc2_vector) {
  KALDI_ASSERT(s2_vector.size() == ilabel_vector.size());

  std::vector<int32> rnn_word_vector;
  std::vector<tensorflow::Tensor*> state_to_context_vector, state_to_cell_vector;
  std::vector<BaseFloat> logprob_vector;
  Label ilabel;
  StateId s;
  int32 rnn_word;
  int parallel_size = s2_vector.size();

  // Scope and Session set up.
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session(root);

  // Build rnn word vector for for GetLogProbParallel.
  for (int iter = 0; iter < parallel_size; ++iter) {
    s = s2_vector[iter];
    ilabel = ilabel_vector[iter];
    state_to_context_vector.push_back(state_to_context_[s]);
    state_to_cell_vector.push_back(state_to_cell_[s]);
    rnn_word = rnnlm_->FstLabelToRnnLabel(ilabel);
    rnn_word_vector.push_back(rnn_word);
  }

  std::vector<Tensor*> state_output_vector, cell_output_vector;
  Tensor state_to_context_tensor, state_to_cell_tensor;

  ConcatTensor(state_to_context_vector, root, session, &state_to_context_tensor);
  ConcatTensor(state_to_cell_vector, root, session, &state_to_cell_tensor);

  Tensor new_context_tensor;
  Tensor new_cell_tensor;
  
  // Get logprob vector.
  rnnlm_->GetLogProbParallel(rnn_word_vector,
                             ilabel_vector,
                             state_to_context_tensor,
                             state_to_cell_tensor,
                             &new_context_tensor,
                             &new_cell_tensor,
                             &logprob_vector);

  KALDI_ASSERT(logprob_vector.size() == parallel_size); 

  std::vector<Tensor> new_context_vector(1), new_cell_vector(1);

  tf_rnnlm::SplitTensor(parallel_size, new_context_tensor, root, session, &new_context_vector);
  tf_rnnlm::SplitTensor(parallel_size, new_cell_tensor, root, session, &new_cell_vector);

  for (int iter = 0; iter < parallel_size; ++iter) {
    s = s2_vector[iter];
    rnn_word = rnn_word_vector[iter];
    Tensor* new_context = new Tensor(new_context_vector[iter]);
    Tensor* new_cell = new Tensor(new_cell_vector[iter]);

    KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());
    std::vector<Label> wseq = state_to_wseq_[s];
    wseq.push_back(rnn_word);

    if (max_ngram_order_ > 0) {
      while (wseq.size() >= max_ngram_order_) {
        // History state has at most <max_ngram_order_> - 1 words in the state.
        wseq.erase(wseq.begin(), wseq.begin() + 1);
      }
    }

    std::pair<const std::vector<Label>, StateId> wseq_state_pair(
        wseq, static_cast<Label>(state_to_wseq_.size()));

    // Attempt to insert the current <wseq_state_pair>. If the pair already
    // exists then it returns false.
    typedef MapType::iterator IterType;
    std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

    // If the pair was just inserted, then also add it to <state_to_wseq_> and
    // <state_to_context_>.
    // In the Parallel case should never fail(?)
    if (result.second == true) {
      state_to_wseq_.push_back(wseq);
      state_to_context_.push_back(new_context);
      state_to_cell_.push_back(new_cell);
    } else {
      delete new_context;
      delete new_cell;
    }

    // Create the arc.
    fst::StdArc arc2;
    arc2.ilabel = ilabel_vector[iter];
    arc2.olabel = ilabel_vector[iter];
    arc2.nextstate = result.first->second;
    arc2.weight = Weight(-logprob_vector[iter]);
    arc2_vector->push_back(arc2);
  } 

} // End of GetArcsParallel.

}  // namespace kaldi

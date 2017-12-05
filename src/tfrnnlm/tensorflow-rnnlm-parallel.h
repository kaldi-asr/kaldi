// tensorflow-rnnlm-lib.h

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

#ifndef KALDI_TFRNNLM_TENSORFLOW_RNNLM_PARALLEL_H_
#define KALDI_TFRNNLM_TENSORFLOW_RNNLM_PARALLEL_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "util/stl-utils.h"
#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tfrnnlm/tensorflow-rnnlm.h"

using tensorflow::Session;
using tensorflow::Tensor;

namespace kaldi {
namespace tf_rnnlm {

class TfRnnlmDeterministicFstParallel:
         public fst::DeterministicOnDemandFstParallel<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  TfRnnlmDeterministicFstParallel(int32 max_ngram_order, KaldiTfRnnlmWrapper *rnnlm);
  ~TfRnnlmDeterministicFstParallel();

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  virtual void FinalParallel(std::vector<StateId> s2_vector_final,
                             std::vector<Weight>* def_fst_final_vector);

  virtual void GetArcsParallel(std::vector<StateId> s1_vector,
                               std::vector<Label> olabel_vector,
                               std::vector<fst::StdArc>* arc2_vector);

  virtual void StackTensor(const std::vector<tensorflow::Input> &input_tensor_vector,
                           const tensorflow::Scope &scope,
                           const tensorflow::ClientSession &session,
                           Tensor *output_tensor);

  virtual void UnstackTensor(int size,
                             const Tensor &input_tensor,
                             const tensorflow::Scope &scope,
                             const tensorflow::ClientSession &session,
                             std::vector<Tensor> *output_tensor_vector);
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

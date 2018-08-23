// rnnlm/rnnlm-core-compute.cc

// Copyright 2017  Daniel Povey

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

#include <numeric>
#include "rnnlm/rnnlm-core-compute.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace rnnlm {


BaseFloat RnnlmCoreComputer::Compute(
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    BaseFloat *weight,
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  using namespace nnet3;

  bool need_model_derivative = false;
  bool need_input_derivative = (word_embedding_deriv != NULL);
  bool store_component_stats = false;

  ComputationRequest request;
  GetRnnlmComputationRequest(minibatch, need_model_derivative,
                             need_input_derivative,
                             store_component_stats,
                             &request);

  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);

  NnetComputeOptions compute_opts;

  NnetComputer computer(compute_opts, *computation, nnet_, NULL);

  ProvideInput(minibatch, derived, word_embedding, &computer);
  computer.Run();  // This is the forward pass.

  BaseFloat ans = ProcessOutput(minibatch, derived, word_embedding,
                                &computer, word_embedding_deriv, weight);

  if (word_embedding_deriv != NULL) {
    computer.Run();  // This is the backward pass.

    CuMatrix<BaseFloat> input_deriv;
    computer.GetOutputDestructive("input", &input_deriv);
    word_embedding_deriv->AddMatSmat(1.0, input_deriv,
                                     derived.input_words_smat,
                                     kTrans, 1.0);
  }
  num_minibatches_processed_++;

  return ans;
}


void RnnlmCoreComputer::ProvideInput(
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    nnet3::NnetComputer *computer) {
  int32 embedding_dim = word_embedding.NumCols();
  CuMatrix<BaseFloat> input_embeddings(derived.cu_input_words.Dim(),
                                       embedding_dim,
                                       kUndefined);
  input_embeddings.CopyRows(word_embedding,
                            derived.cu_input_words);
  computer->AcceptInput("input", &input_embeddings);
}


RnnlmCoreComputer::RnnlmCoreComputer(const nnet3::Nnet &nnet):
    nnet_(nnet),
    compiler_(nnet),  // for now we don't make available other options
    num_minibatches_processed_(0),
    objf_info_(10) { }

BaseFloat RnnlmCoreComputer::ProcessOutput(
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    nnet3::NnetComputer *computer,
    CuMatrixBase<BaseFloat> *word_embedding_deriv,
    BaseFloat *weight_out) {
  // 'output' is the output of the neural network.  The row-index
  // combines the time (with higher stride) and the member 'n'
  // of the minibatch (with stride 1); the number of columns is
  // the word-embedding dimension.
  CuMatrix<BaseFloat> output;
  CuMatrix<BaseFloat> output_deriv;
  computer->GetOutputDestructive("output", &output);
  output_deriv.Resize(output.NumRows(), output.NumCols());

  BaseFloat weight, objf_num, objf_den, objf_den_exact;


  RnnlmObjectiveOptions objective_opts;  // Use the defaults; we're not training
                                         // so they won't matter.
  ProcessRnnlmOutput(objective_opts, minibatch, derived, word_embedding,
                     output, word_embedding_deriv, &output_deriv,
                     &weight, &objf_num, &objf_den,
                     &objf_den_exact);

  objf_info_.AddStats(weight, objf_num, objf_den, objf_den_exact);
  if (weight_out)
    *weight_out = weight;
  return objf_num + objf_den;
}




}  // namespace rnnlm
}  // namespace kaldi

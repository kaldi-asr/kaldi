// nnet3/nnet-cctc-diagnostics.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-cctc-diagnostics.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetCctcComputeProb::NnetCctcComputeProb(
    const NnetCctcComputeProbOptions &config,
    const ctc::CctcTransitionModel &trans_model,
    const Nnet &nnet):
    config_(config),
    trans_model_(trans_model),
    nnet_(nnet),
    deriv_nnet_(NULL),
    compiler_(nnet),
    num_minibatches_processed_(0) {
  if (config_.compute_deriv) {
    deriv_nnet_ = new Nnet(nnet_);
    bool is_gradient = true;  // force simple update
    SetZero(is_gradient, deriv_nnet_);
  }
  trans_model_.ComputeWeights(&cu_weights_);
}

const Nnet &NnetCctcComputeProb::GetDeriv() const {
  if (deriv_nnet_ == NULL)
    KALDI_ERR << "GetDeriv() called when no derivatives were requested.";
  return *deriv_nnet_;
}

NnetCctcComputeProb::~NnetCctcComputeProb() {
  delete deriv_nnet_;  // delete does nothing if pointer is NULL.
}

void NnetCctcComputeProb::Reset() {
  num_minibatches_processed_ = 0;
  objf_info_.clear();
  if (deriv_nnet_) {
    bool is_gradient = true;
    SetZero(is_gradient, deriv_nnet_);
  }
}

void NnetCctcComputeProb::Compute(const NnetCctcExample &cctc_eg) {
  bool need_model_derivative = config_.compute_deriv,
      store_component_stats = false;
  ComputationRequest request;
  GetCctcComputationRequest(nnet_, cctc_eg, need_model_derivative,
                            store_component_stats,
                            &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, cctc_eg.inputs);
  computer.Forward();
  this->ProcessOutputs(cctc_eg, &computer);
  if (config_.compute_deriv)
    computer.Backward();
}

void NnetCctcComputeProb::ProcessOutputs(const NnetCctcExample &eg,
                                         NnetComputer *computer) {
  // There will normally be just one output here, named 'output',
  // but the code is more general than this.
  std::vector<NnetCctcSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetCctcSupervision &sup = *iter;
    int32 node_index = nnet_.GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_.IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;
    
    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv;
    if (config_.compute_deriv)
      nnet_output_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                               kUndefined);
    
    BaseFloat tot_weight, tot_objf;
    sup.ComputeObjfAndDerivs(config_.cctc_training_config,
                             trans_model_,
                             cu_weights_, nnet_output,
                             &tot_weight, &tot_objf,
                             (config_.compute_deriv ?
                              &nnet_output_deriv : NULL));

    SimpleObjectiveInfo &totals = objf_info_[sup.name];
    totals.tot_weight += tot_weight;
    totals.tot_objective += tot_objf;
    
    if (config_.compute_deriv)
      computer->AcceptOutputDeriv(sup.name, &nnet_output_deriv);

    num_minibatches_processed_++;
  }
}

bool NnetCctcComputeProb::PrintTotalStats() const {
  bool ans = false;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter, end;
  iter = objf_info_.begin();
  end = objf_info_.end();
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    int32 node_index = nnet_.GetNodeIndex(name);
    KALDI_ASSERT(node_index >= 0);
    const SimpleObjectiveInfo &info = iter->second;
    KALDI_LOG << "Overall log-probability for '"
              << name << "' is "
              << (info.tot_objective / info.tot_weight) << " per frame"
              << ", over " << info.tot_weight << " frames.";
    if (info.tot_weight > 0)
      ans = true;
  }
  return ans;
}


const SimpleObjectiveInfo* NnetCctcComputeProb::GetObjective(
    const std::string &output_name) const {
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter = objf_info_.find(output_name);
  if (iter != objf_info_.end())
    return &(iter->second);
  else
    return NULL;
}

} // namespace nnet3
} // namespace kaldi

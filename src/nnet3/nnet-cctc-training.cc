// nnet3/nnet-training.cc

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

#include "nnet3/nnet-cctc-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetCctcTrainer::NnetCctcTrainer(const NnetCctcTrainerOptions &config,
                                 const ctc::CctcTransitionModel &trans_model,
                                 Nnet *nnet):
    config_(config),
    trans_model_(trans_model),
    nnet_(nnet),
    compiler_(*nnet, config_.optimize_config),
    num_minibatches_processed_(0) {
  if (config.zero_component_stats)
    ZeroComponentStats(nnet);
  if (config.momentum == 0.0 && config.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(config.momentum >= 0.0 &&
                 config.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    SetZero(is_gradient, delta_nnet_);
  }
  trans_model_.ComputeWeights(&cu_weights_);
}


void NnetCctcTrainer::Train(const NnetCctcExample &cctc_eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
  GetCctcComputationRequest(*nnet_, cctc_eg, need_model_derivative,
                            config_.store_component_stats,
                            &request);
  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(config_.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, cctc_eg.inputs);
  computer.Forward();

  this->ProcessOutputs(cctc_eg, &computer);
  computer.Backward();

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - config_.momentum);
    if (config_.max_param_change != 0.0) {
      BaseFloat param_delta =
          std::sqrt(DotProduct(*delta_nnet_, *delta_nnet_)) * scale;
      if (param_delta > config_.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          SetZero(false, delta_nnet_);
        } else {
          scale *= config_.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << config_.max_param_change
                    << ", scaling by " << config_.max_param_change / param_delta;
        }
      }
    }
    AddNnet(*delta_nnet_, scale, nnet_);
    ScaleNnet(config_.momentum, delta_nnet_);
  }
}


void NnetCctcTrainer::ProcessOutputs(const NnetCctcExample &eg,
                                     NnetComputer *computer) {
  // There will normally be just one output here, named 'output',
  // but the code is more general than this.
  std::vector<NnetCctcSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetCctcSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    BaseFloat tot_weight, tot_num_objf, tot_den_objf;
    sup.ComputeObjfAndDerivs(config_.cctc_training_config,
                             trans_model_,
                             cu_weights_, nnet_output,
                             &tot_weight, &tot_num_objf, &tot_den_objf,
                             &nnet_output_deriv);

    computer->AcceptOutputDeriv(sup.name, &nnet_output_deriv);

    objf_info_[sup.name].UpdateStats(sup.name, config_.print_interval,
                                     num_minibatches_processed_++,
                                     tot_weight, tot_num_objf, tot_den_objf);
  }
}


bool NnetCctcTrainer::PrintTotalStats() const {
  unordered_map<std::string, CctcObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const CctcObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}


NnetCctcTrainer::~NnetCctcTrainer() {
  delete delta_nnet_;
}


CctcObjectiveFunctionInfo::CctcObjectiveFunctionInfo():
    current_phase(0),
    tot_weight(0.0), tot_num_objf(0.0), tot_den_objf(0.0),
    tot_weight_this_phase(0.0),
    tot_num_objf_this_phase(0.0),
    tot_den_objf_this_phase(0.0) { }


void CctcObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    BaseFloat this_minibatch_weight,
    BaseFloat this_minibatch_tot_num_objf,
    BaseFloat this_minibatch_tot_den_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase == current_phase + 1); // or doesn't really make sense.
    PrintStatsForThisPhase(output_name, minibatches_per_phase);
    current_phase = phase;
    tot_weight_this_phase = 0.0;
    tot_num_objf_this_phase = 0.0;
    tot_den_objf_this_phase = 0.0;
  }
  tot_weight_this_phase += this_minibatch_weight;
  tot_num_objf_this_phase += this_minibatch_tot_num_objf;
  tot_den_objf_this_phase += this_minibatch_tot_den_objf;
  tot_weight += this_minibatch_weight;
  tot_num_objf += this_minibatch_tot_num_objf;
  tot_den_objf += this_minibatch_tot_den_objf;
}

void CctcObjectiveFunctionInfo::PrintStatsForThisPhase(
    const std::string &output_name,
    int32 minibatches_per_phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = start_minibatch + minibatches_per_phase - 1;
  BaseFloat tot_objf_this_phase = tot_num_objf_this_phase +
      tot_den_objf_this_phase;
  KALDI_LOG << "Average objective function for '" << output_name
            << "' for minibatches " << start_minibatch
            << '-' << end_minibatch << " is "
            << (tot_objf_this_phase / tot_weight_this_phase) << " over "
            << tot_weight_this_phase << " frames = "
            << (tot_num_objf_this_phase / tot_weight_this_phase) << " + "
            << (tot_den_objf_this_phase / tot_weight_this_phase);
}

bool CctcObjectiveFunctionInfo::PrintTotalStats(const std::string &name) const {
  BaseFloat tot_objf = tot_num_objf + tot_den_objf;
  KALDI_LOG << "Overall average objective function for '" << name << "' is "
            << (tot_objf / tot_weight) << " over " << tot_weight << " frames = "
            << (tot_num_objf / tot_weight) << " + "
            << (tot_den_objf / tot_weight);

  KALDI_LOG << "[this line is to be parsed by a script:] "
            << "log-prob-per-frame="
            << (tot_objf / tot_weight);
  return (tot_weight != 0.0);
}


} // namespace nnet3
} // namespace kaldi

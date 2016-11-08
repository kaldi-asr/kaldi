
// Copyright      2016 Hainan Xu

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

#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-utils.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

LmNnetTrainer::LmNnetTrainer(const LmNnetTrainerOptions &config,
                             LmNnet *nnet):
    config_(config),
    nnet_(nnet),
    compiler_(*nnet->GetNnet(), config_.optimize_config),
    num_minibatches_processed_(0)
//    input_projection_(config_.input_dim, config_.nnet_input_dim),
//    output_projection_(config_.output_dim, config_.nnet_output_dim)
  {
  if (config.zero_component_stats)
    ZeroComponentStats(nnet->GetNnet());
    nnet_->I()->ZeroStats();
    nnet_->O()->ZeroStats();
  if (config.momentum == 0.0 && config.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(config.momentum >= 0.0 &&
                 config.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    SetZero(is_gradient, delta_nnet_->GetNnet());
    delta_nnet_->I()->SetZero(is_gradient);
    delta_nnet_->O()->SetZero(is_gradient);
  }
  if (config_.read_cache != "") {
    bool binary;
    try {
      Input ki(config_.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << config_.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  } 
}

NnetExample LmNnetTrainer::ProcessEgInputs(NnetExample eg,
                                           const AffineComponent& a) {
  for (size_t i = 0; i < eg.io.size(); i++) {
    NnetIo &io = eg.io[i];

    if (io.name == "input") {
      new_input_.Resize(io.features.NumRows(),
                        a.OutputDim(),
                        kUndefined);
      old_input_.Resize(io.features.NumRows(),
                        io.features.NumCols(),
                        kUndefined);
      old_input_.CopyFromGeneralMat(io.features);

      a.Propagate(NULL, old_input_, &new_input_);
      //        SparseMatrix<BaseFloat> sp = io.features.GetSparseMatrix();
      //
      //        for (size_t i = 0; i < sp.NumRows(); i++) {
      //          SparseVector<BaseFloat> sv = sp.Row(i);
      //          int non_zero_index = -1;
      //          sv.Max(&non_zero_index);
      ////          cu_input.CopyRows(projection.RowData(non_zero_index));
      //          cu_input.CopyRowsFromVec(projection.Row(non_zero_index));
      //        }
      Matrix<BaseFloat> input(new_input_);
      io.features = input;
    }
  }
  return eg;
}

void LmNnetTrainer::Train(const NnetExample &eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
  GetComputationRequest(*nnet_->GetNnet(), eg, need_model_derivative,
                        config_.store_component_stats,
                        &request);

  KALDI_ASSERT(request.inputs.size() == 1);
  request.inputs[0].has_deriv = true;

  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(config_.compute_config, *computation,
                        *nnet_->GetNnet(),
                        (delta_nnet_ == NULL ? nnet_->GetNnet() :
                               delta_nnet_->GetNnet()));

  NnetExample new_eg = ProcessEgInputs(eg, *nnet_->I());

  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_->GetNnet(), new_eg.io);
  computer.Forward();

  // in ProcessOutputs() we first do the last Forward propagation
  // and before exiting, do the first step of back-propagation
  this->ProcessOutputs(eg, &computer);
  computer.Backward();

  {
    CuMatrix<BaseFloat> first_deriv(computer.GetInputDeriv("input"));

    CuMatrix<BaseFloat> place_holder;

    nnet_->I()->Backprop("", NULL, old_input_, place_holder,
                     first_deriv, delta_nnet_->I(), NULL);

  }

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - config_.momentum);
    if (config_.max_param_change != 0.0) {
      BaseFloat param_delta =
          DotProduct(*delta_nnet_->GetNnet(), *delta_nnet_->GetNnet());
//      KALDI_LOG << "param_delta currently " << param_delta;
      param_delta += delta_nnet_->I()->DotProduct(*delta_nnet_->I());
//      KALDI_LOG << "param_delta currently " << param_delta;
      param_delta += delta_nnet_->O()->DotProduct(*delta_nnet_->O());
//      KALDI_LOG << "param_delta currently " << param_delta;

      param_delta = std::sqrt(param_delta) * scale;
//          std::sqrt(DotProduct(*delta_nnet_->GetNnet(), *delta_nnet_->GetNnet())) * scale;
      if (param_delta > config_.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          SetZero(false, delta_nnet_->GetNnet());
          delta_nnet_->I()->SetZero(false);
          delta_nnet_->O()->SetZero(false);
        } else {
          scale *= config_.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << config_.max_param_change
                    << ", scaling by " << config_.max_param_change / param_delta;
        }
      }
    }
    AddNnet(*delta_nnet_->GetNnet(), scale, nnet_->GetNnet());
    nnet_->I()->Add(scale, *delta_nnet_->I());
    nnet_->O()->Add(scale, *delta_nnet_->O());

    ScaleNnet(config_.momentum, delta_nnet_->GetNnet());
    delta_nnet_->I()->Scale(config_.momentum);
    delta_nnet_->O()->Scale(config_.momentum);
  }
}

void LmNnetTrainer::ProcessOutputs(const NnetExample &eg,
                                   NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_->GetNnet()->GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->GetNnet()->IsOutputNode(node_index)) {
      ObjectiveType obj_type = nnet_->GetNnet()->GetNode(node_index).u.objective_type;
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;

      // the following function adds the computation of special layers
      ComputeObjectiveFunction(io.features, obj_type, io.name,
                               supply_deriv, computer,
                               &tot_weight, &tot_objf, nnet_->O(), nnet_->N(), delta_nnet_);

      objf_info_[io.name].UpdateStats(io.name, config_.print_interval,
                                      num_minibatches_processed_++,
                                      tot_weight, tot_objf);
    }
  }
}

bool LmNnetTrainer::PrintTotalStats() const {
  unordered_map<std::string, LmObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const LmObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}

void LmObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    BaseFloat this_minibatch_weight,
    BaseFloat this_minibatch_tot_objf,
    BaseFloat this_minibatch_tot_aux_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase == current_phase + 1); // or doesn't really make sense.
    PrintStatsForThisPhase(output_name, minibatches_per_phase);
    current_phase = phase;
    tot_weight_this_phase = 0.0;
    tot_objf_this_phase = 0.0;
    tot_aux_objf_this_phase = 0.0;
  }
  tot_weight_this_phase += this_minibatch_weight;
  tot_objf_this_phase += this_minibatch_tot_objf;
  tot_aux_objf_this_phase += this_minibatch_tot_aux_objf;
  tot_weight += this_minibatch_weight;
  tot_objf += this_minibatch_tot_objf;
  tot_aux_objf += this_minibatch_tot_aux_objf;
}

void LmObjectiveFunctionInfo::PrintStatsForThisPhase(
    const std::string &output_name,
    int32 minibatches_per_phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = start_minibatch + minibatches_per_phase - 1;

  if (tot_aux_objf_this_phase == 0.0) {
    KALDI_LOG << "Average objective function for '" << output_name
              << "' for minibatches " << start_minibatch
              << '-' << end_minibatch << " is "
              << (tot_objf_this_phase / tot_weight_this_phase) << " over "
              << tot_weight_this_phase << " frames.";
  } else {
    BaseFloat objf = (tot_objf_this_phase / tot_weight_this_phase),
        aux_objf = (tot_aux_objf_this_phase / tot_weight_this_phase),
        sum_objf = objf + aux_objf;
    KALDI_LOG << "Average objective function for '" << output_name
              << "' for minibatches " << start_minibatch
              << '-' << end_minibatch << " is "
              << objf << " + " << aux_objf << " = " << sum_objf
              << " over " << tot_weight_this_phase << " frames.";
  }
}

bool LmObjectiveFunctionInfo::PrintTotalStats(const std::string &name) const {
  BaseFloat objf = (tot_objf / tot_weight),
        aux_objf = (tot_aux_objf / tot_weight),
        sum_objf = objf + aux_objf;
  if (tot_aux_objf == 0.0) {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << (tot_objf / tot_weight) << " over " << tot_weight << " frames.";
  } else {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << objf << " + " << aux_objf << " = " << sum_objf        
              << " over " << tot_weight << " frames.";
  }
  KALDI_LOG << "[this line is to be parsed by a script:] "
            << "log-prob-per-frame="
            << objf;
  return (tot_weight != 0.0);
}

LmNnetTrainer::~LmNnetTrainer() {
  if (config_.write_cache != "") {
    Output ko(config_.write_cache, config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << config_.write_cache;
  } 
  delete delta_nnet_;
}

CuMatrix<BaseFloat> ProcessOutput(const CuMatrixBase<BaseFloat> &output_0,
                                  const Component *output_projection_1,
                                  const Component *output_projection_2) {
  CuMatrix<BaseFloat> ans(output_0.NumRows(), output_projection_1->OutputDim());
  output_projection_1->Propagate(NULL, output_0, &ans);
  output_projection_2->Propagate(NULL, ans, &ans);

  return ans;
}

void ComputeObjectiveFunction(const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf,
                              const Component *output_projection_1,
                              const Component *output_projection_2,
                              LmNnet *nnet) {
  const CuMatrixBase<BaseFloat> &output_0 = computer->GetOutput(output_name);

  const CuMatrix<BaseFloat> &output =
              ProcessOutput(output_0, output_projection_1, output_projection_2); 

  if (output.NumCols() != supervision.NumCols())
    KALDI_ERR << "Nnet versus example output dimension (num-classes) "
              << "mismatch for '" << output_name << "': " << output.NumCols()
              << " (nnet) vs. " << supervision.NumCols() << " (egs)\n";

  switch (objective_type) {
    case kLinear: {
      // objective is x * y.
      switch (supervision.Type()) {
        case kSparseMatrix: {
          const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();
          CuSparseMatrix<BaseFloat> cu_post(post);
          // The cross-entropy objective is computed by a simple dot product,
          // because after the LogSoftmaxLayer, the output is already in the form
          // of log-likelihoods that are normalized to sum to one.
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatSmat(output, cu_post, kTrans);
          if (supply_deriv) {
            // the derivative on the real output
            CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols(),
                                             kSetZero);

            // the derivative after the affine layer (before the nonlin)
            CuMatrix<BaseFloat> between_deriv(output.NumRows(), output.NumCols(),
                                              kSetZero);

            // the derivative of the 'nnet3' part
            CuMatrix<BaseFloat> input_deriv(output.NumRows(), output_0.NumCols(),
                                            kSetZero);

            cu_post.CopyToMat(&output_deriv);
            CuMatrix<BaseFloat> place_holder;
            output_projection_2->Backprop("", NULL, place_holder, output,
                             output_deriv, NULL, &between_deriv);

            output_projection_1->Backprop("", NULL, output_0, place_holder,
                             between_deriv, nnet->O(), &input_deriv);

            computer->AcceptOutputDeriv(output_name, &input_deriv);
          }
          break;
        }
        default: {
                   KALDI_ASSERT(false);
                 }
//        case kFullMatrix: {
//          // there is a redundant matrix copy in here if we're not using a GPU
//          // but we don't anticipate this code branch being used in many cases.
//          CuMatrix<BaseFloat> cu_post(supervision.GetFullMatrix());
//          *tot_weight = cu_post.Sum();
//          *tot_objf = TraceMatMat(output, cu_post, kTrans);
//          if (supply_deriv)
//            computer->AcceptOutputDeriv(output_name, &cu_post);
//          break;
//        }
//        case kCompressedMatrix: {
//          Matrix<BaseFloat> post;
//          supervision.GetMatrix(&post);
//          CuMatrix<BaseFloat> cu_post;
//          cu_post.Swap(&post);
//          *tot_weight = cu_post.Sum();
//          *tot_objf = TraceMatMat(output, cu_post, kTrans);
//          if (supply_deriv)
//            computer->AcceptOutputDeriv(output_name, &cu_post);
//          break;
//        }
      }
      break;
    }
    case kQuadratic: {
      // objective is -0.5 (x - y)^2
      CuMatrix<BaseFloat> diff(supervision.NumRows(),
                               supervision.NumCols(),
                               kUndefined);
      diff.CopyFromGeneralMat(supervision);
      diff.AddMat(-1.0, output);
      *tot_weight = diff.NumRows();
      *tot_objf = -0.5 * TraceMatMat(diff, diff, kTrans);
      if (supply_deriv)
        computer->AcceptOutputDeriv(output_name, &diff);
      break;
    }
    default:
      KALDI_ERR << "Objective function type " << objective_type
                << " not handled.";
  }
}



} // namespace nnet3
} // namespace kaldi

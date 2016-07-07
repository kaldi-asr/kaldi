// nnet3/nnet-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2015    Xiaohui Zhang

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

#include "nnet3/nnet-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetTrainer::NnetTrainer(const NnetTrainerOptions &config,
                         Nnet *nnet):
    config_(config),
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


void NnetTrainer::Train(const NnetExample &eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
  GetComputationRequest(*nnet_, eg, need_model_derivative,
                        config_.store_component_stats,
                        &request);
  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(config_.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.io);
  Timer tim;//debug
  computer.Forward();
  KALDI_LOG << "forward time: " << tim.Elapsed();//debug

  tim.Reset();//debug
  this->ProcessOutputs(eg, &computer);
  KALDI_LOG << "processoutputs time: " << tim.Elapsed();//debug
  // Update the recurrent output matrices in recurrent_outputs_
  // as additional inputs for the next minibatch, if there is any
  tim.Reset();//debug
  UpdateRecurrentOutputs(eg, computer);
  KALDI_LOG << "update time: " << tim.Elapsed();//debug
  tim.Reset();//debug
  computer.Backward();
  KALDI_LOG << "backward time: " << tim.Elapsed();//debug

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

void NnetTrainer::ProcessOutputs(const NnetExample &eg,
                                 NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_->GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->IsOutputNode(node_index)) {
      ObjectiveType obj_type = nnet_->GetNode(node_index).u.objective_type;
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;
      ComputeObjectiveFunction(io.features, obj_type, io.name,
                               supply_deriv, computer,
                               &tot_weight, &tot_objf);
      objf_info_[io.name].UpdateStats(io.name, config_.print_interval,
                                      io.name == "output" ?
                                      num_minibatches_processed_++ :
                                      num_minibatches_processed_,
                                      tot_weight, tot_objf);
    }
  }
}

void NnetTrainer::GiveStatePreservingInfo(const std::vector<std::string>
                                          &recurrent_output_names,
                                          const std::vector<int32>
                                          &recurrent_offsets) {
  recurrent_output_names_ = recurrent_output_names;
  recurrent_offsets_ = recurrent_offsets;
  recurrent_outputs_.resize(recurrent_output_names.size());
}

void NnetTrainer::UpdateRecurrentOutputs(const NnetExample &eg,
                                         const NnetComputer &computer) {
  //for (int32 i = 0; i < recurrent_output_names_.size(); i++) {//debug
    //KALDI_ASSERT(recurrent_output_names_[i] + "_STATE_PREVIOUS_MINIBATCH" == eg.io[i * 2 + 3].name);//debug
    //CuMatrix<BaseFloat> feat(eg.io[i * 2+ 3].features.NumRows(), eg.io[i * 2 + 3].features.NumCols(), kUndefined);//debug
    //eg.io[i * 2 + 3].features.CopyToMat(&feat);//debug
    //if (feat.FrobeniusNorm() < 0.0001) {//debug
      //KALDI_ASSERT(ApproxEqual(recurrent_outputs_[i], feat, static_cast<BaseFloat>(0.0001)));//debug
    //}//debug
  //}//debug
  // compute the chunk size and num of chunks of the current minibatch
  int32 chunk_size = 0, num_chunks = 0;
  for (int32 f = 0; f < eg.io.size(); f++) {
    if (eg.io[f].name == "output") {
      chunk_size = NumFramesPerChunk(eg.io[f]);
      num_chunks = NumChunks(eg.io[f]);
      break;
    }
  }

  for (int32 i = 0; i < recurrent_output_names_.size(); i++) {
    const std::string &node_name = recurrent_output_names_[i];
    // get the cuda matrix corresponding to the recurrent output
    const CuMatrixBase<BaseFloat> &r_all_cu = computer.GetOutput(node_name);
    KALDI_ASSERT(r_all_cu.NumRows() == num_chunks * chunk_size);

    // only copy the rows corresponding to the recurrent output of the
    // last (if offset < 0) or first (if offset > 0) [abs(offset)] frames 
    // of each chunk from the previous minibatch
    const int32 offset = recurrent_offsets_[i];
    KALDI_ASSERT(offset != 0);
    std::vector<int32> indexes(num_chunks * abs(offset));
    for (int32 n = 0; n < num_chunks; n++) {
      for (int32 t = 0; t < abs(offset); t++) {
        if (offset < 0)
          indexes[n * abs(offset) + t] = (n + 1) * chunk_size + offset + t;
        else
          indexes[n * abs(offset) + t] = n * chunk_size + t;
      }
    }

    CuArray<int32> indexes_cu(indexes);

    recurrent_outputs_[i].Resize(num_chunks * abs(offset), r_all_cu.NumCols(),
                                 kUndefined);
    recurrent_outputs_[i].CopyRows(r_all_cu, indexes_cu);
    //for (int32 n = 0; n < num_chunks; n++) {//debug
      //for (int32 t = 0; t < abs(offset); t++) { //debug
        //if (offset < 0) //debug 
          //AssertEqual(r_all_cu.RowRange((n + 1) * chunk_size + offset, abs(offset)),//debug
            //          recurrent_outputs_[i].RowRange(n * abs(offset), abs(offset)),//debug
              //        static_cast<BaseFloat>(0.0001));//debug
        //else //debug
          //AssertEqual(r_all_cu.RowRange(n * chunk_size, abs(offset)),//debug
            //          recurrent_outputs_[i].RowRange(n * abs(offset), abs(offset)),//debug
              //        static_cast<BaseFloat>(0.0001));//debug 
      //}//debug
    //}//debug
  }
}

bool NnetTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}

void ObjectiveFunctionInfo::UpdateStats(
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

void ObjectiveFunctionInfo::PrintStatsForThisPhase(
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

bool ObjectiveFunctionInfo::PrintTotalStats(const std::string &name) const {
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

NnetTrainer::~NnetTrainer() {
  if (config_.write_cache != "") {
    Output ko(config_.write_cache, config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << config_.write_cache;
  } 
  delete delta_nnet_;
}

void ComputeObjectiveFunction(const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf) {
  const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name);

  // create zero matrix as output derivative of the recurrent output node
  int32 node_index = computer->GetNnet().GetNodeIndex(output_name +
                     "_STATE_PREVIOUS_MINIBATCH");
  if (node_index != -1 && computer->GetNnet().IsInputNode(node_index)) {
    *tot_weight = 0;
    *tot_objf = 0;
    if (supply_deriv) {
      CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
      computer->AcceptOutputDeriv(output_name, &output_deriv);
    }
    return;
  }

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
            CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols(),
                                             kUndefined);
            cu_post.CopyToMat(&output_deriv);
            computer->AcceptOutputDeriv(output_name, &output_deriv);
          }
          break;
        }
        case kFullMatrix: {
          // there is a redundant matrix copy in here if we're not using a GPU
          // but we don't anticipate this code branch being used in many cases.
          CuMatrix<BaseFloat> cu_post(supervision.GetFullMatrix());
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptOutputDeriv(output_name, &cu_post);
          break;
        }
        case kCompressedMatrix: {
          Matrix<BaseFloat> post;
          supervision.GetMatrix(&post);
          CuMatrix<BaseFloat> cu_post;
          cu_post.Swap(&post);
          *tot_weight = cu_post.Sum();
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptOutputDeriv(output_name, &cu_post);
          break;
        }
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

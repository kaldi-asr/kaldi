// xvector/nnet-xvector-diagnostics.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
// Copyright      2016    Pegah Ghahremani
//                        David Snyder
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

#include "nnet3/nnet-diagnostics.h"
#include "xvector/nnet-xvector-diagnostics.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetXvectorComputeProb::NnetXvectorComputeProb(const NnetComputeProbOptions
                                               &config,
                                               const Nnet &nnet):
    config_(config),
    nnet_(nnet),
    deriv_nnet_(NULL),
    compiler_(nnet),
    num_minibatches_processed_(0) {
  if (config_.compute_deriv) {
    deriv_nnet_ = new Nnet(nnet_);
    ScaleNnet(0.0, deriv_nnet_); // force simple update
    SetNnetAsGradient(deriv_nnet_);
  }
}

const Nnet &NnetXvectorComputeProb::GetDeriv() const {
  if (deriv_nnet_ == NULL)
    KALDI_ERR << "GetDeriv() called when no derivatives were requested.";
  return *deriv_nnet_;
}

NnetXvectorComputeProb::~NnetXvectorComputeProb() {
  delete deriv_nnet_;  // delete does nothing if pointer is NULL.
}

void NnetXvectorComputeProb::Reset() {
  num_minibatches_processed_ = 0;
  objf_info_.clear();
  if (deriv_nnet_) {
    ScaleNnet(0.0, deriv_nnet_);
    SetNnetAsGradient(deriv_nnet_);
  }
}

void NnetXvectorComputeProb::Compute(const NnetExample &eg) {
  bool need_model_derivative = config_.compute_deriv,
      store_component_stats = false;
  ComputationRequest request;
  GetComputationRequestXvector(nnet_, eg, need_model_derivative,
                               store_component_stats,
                               &request);
  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);
  NnetComputer computer(config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, eg.io);
  computer.Run();
  this->ProcessOutputs(&computer);
  if (config_.compute_deriv)
    computer.Run();
}

void NnetXvectorComputeProb::ProcessOutputs(NnetComputer *computer) {
  for (int32 node_index = 0; node_index < nnet_.NumNodes(); node_index++) {
    if (nnet_.IsOutputNode(node_index)) {
      std::string xvector_name = nnet_.GetNodeName(node_index),
        s_name = "s", b_name = "b";
      if (nnet_.GetNodeIndex(s_name) == -1 || nnet_.GetNodeIndex(b_name) == -1)
        KALDI_ERR << "The nnet expected to have two output nodes with "
                  << "name s and b.";

      if (xvector_name == "output") {
        const CuMatrixBase<BaseFloat> &xvector_pairs
                                      = computer->GetOutput(xvector_name),
                                      &xvec_s = computer->GetOutput(s_name),
                                      &xvec_b = computer->GetOutput(b_name);
        int32 num_rows = xvector_pairs.NumRows(),
              num_cols = xvector_pairs.NumCols();
        CuMatrix<BaseFloat> xvector_deriv(num_rows, num_cols, kUndefined),
                            raw_scores(num_rows, num_rows, kUndefined);
        int32 s_dim = num_cols * (num_cols + 1) / 2;

        // convert CuVector to CuSpMatrix
        CuSpMatrix<BaseFloat> xvec_s_sp(num_cols);
        xvec_s_sp.CopyFromVec(xvec_s.Row(0));
        CuVector<BaseFloat> deriv_s(s_dim);

        BaseFloat xvec_b_val = xvec_b(0,0), deriv_b;
        BaseFloat tot_weight, tot_objf;
        bool supply_deriv = config_.compute_deriv;
        bool compute_accuracy = config_.compute_accuracy;
        ComputeXvectorObjfAndDeriv(xvector_pairs, xvec_s_sp, xvec_b_val,
                                   (supply_deriv ? &xvector_deriv : NULL),
                                   (supply_deriv ? &deriv_s : NULL),
                                   (supply_deriv ? &deriv_b : NULL),
                                   (compute_accuracy ? &raw_scores : NULL),
                                   &tot_objf,
                                   &tot_weight);
        if (supply_deriv) {
          CuMatrix<BaseFloat> deriv_s_mat(1, s_dim),
            deriv_b_mat(1,1);
          deriv_b_mat(0,0) = deriv_b;
          deriv_s_mat.CopyRowsFromVec(deriv_s);
          computer->AcceptInput(xvector_name, &xvector_deriv);
          computer->AcceptInput(s_name, &deriv_s_mat);
          computer->AcceptInput(b_name, &deriv_b_mat);

        }
        SimpleObjectiveInfo &totals = objf_info_[xvector_name];
        totals.tot_weight += tot_weight;
        totals.tot_objective += tot_objf;
        if (compute_accuracy) {
          BaseFloat tot_acc;
          SimpleObjectiveInfo &acc_totals = acc_info_[xvector_name];
          ComputeAccuracy(raw_scores, &tot_acc);
          acc_totals.tot_objective += tot_weight * tot_acc;
          acc_totals.tot_weight += tot_weight;
        }
      }
      num_minibatches_processed_++;
    }
  }
}

bool NnetXvectorComputeProb::PrintTotalStats() const {
  bool ans = false;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter, end;
  { // First print regular objectives
    iter = objf_info_.begin();
    end = objf_info_.end();
    for (; iter != end; ++iter) {
      const std::string &name = iter->first;
      int32 node_index = nnet_.GetNodeIndex(name);
      KALDI_ASSERT(node_index >= 0);
      ObjectiveType obj_type = nnet_.GetNode(node_index).u.objective_type;
      const SimpleObjectiveInfo &info = iter->second;
      KALDI_LOG << "Overall "
                << (obj_type == kLinear ? "log-likelihood" : "objective")
                << " for '" << name << "' is "
                << (info.tot_objective / info.tot_weight) << " per chunk"
                << ", over " << info.tot_weight << " chunks.";
      if (info.tot_weight > 0)
        ans = true;
    }
  }
  if (config_.compute_accuracy) {  // Now print the accuracy.
    iter = acc_info_.begin();
    end = acc_info_.end();
    for (; iter != end; ++iter) {
      const std::string &name = iter->first;
      const SimpleObjectiveInfo &info = iter->second;
      KALDI_LOG << "Overall accuracy for '" << name << "' is "
                << (info.tot_objective / info.tot_weight)
                << " per chunk"
                << ", over " << ceil(info.tot_weight) << " chunks.";
    }
  }
  return ans;
}

void NnetXvectorComputeProb::ComputeAccuracy(
    const CuMatrixBase<BaseFloat> &raw_scores,
    BaseFloat *tot_accuracy_out) {
  int32 num_rows = raw_scores.NumRows();
  BaseFloat K = 1.0 / (num_rows - 2.0),
            threshold = 0; // Corresponds to prob_same(u,v) = 0.5.
  BaseFloat count = 0,
        error = 0;
  for (int32 i = 0; i < num_rows; i++) {
    for (int32 j = 0; j < num_rows; j++) {
      if (i + 1 == j && i % 2 == 0) {
        if (raw_scores(i, j) < threshold)
          error++;
        count++;
      } else if (i < j) {
        if (raw_scores(i, j) >= threshold)
          error += K;
        count += K;
      }
    }
  }
  (*tot_accuracy_out) = 1.0 - error / count;
}

const SimpleObjectiveInfo* NnetXvectorComputeProb::GetObjective(
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

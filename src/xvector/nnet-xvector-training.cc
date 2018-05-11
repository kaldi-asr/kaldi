// xvector/nnet-xvector-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2015    Xiaohui Zhang
// Copyright      2016    Pegah Ghahremani
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

#include "xvector/nnet-xvector-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetXvectorTrainer::NnetXvectorTrainer(const NnetTrainerOptions &config,
                         Nnet *nnet):
    config_(config),
    nnet_(nnet),
    compiler_(*nnet, config_.optimize_config),
    num_minibatches_processed_(0) {
  if (config_.zero_component_stats)
    ZeroComponentStats(nnet);
  if (config_.momentum == 0.0 &&
      config_.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(config_.momentum >= 0.0 &&
                 config_.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    if (is_gradient) {
      ScaleNnet(0.0, delta_nnet_);
      SetNnetAsGradient(delta_nnet_);
    } else {
      ScaleNnet(0.0, delta_nnet_);
    }
  }
  if (config_.read_cache != "") {
    bool binary;
    try {
      Input ki(config_.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }
}


void NnetXvectorTrainer::Train(const NnetExample &eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
  GetComputationRequestXvector(*nnet_, eg, need_model_derivative,
                               config_.store_component_stats,
                               &request);
  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);

  NnetComputer computer(config_.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.io);
  computer.Run();

  this->ProcessOutputs(&computer);
  computer.Run();

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - config_.momentum);
    if (config_.max_param_change != 0.0) {
      BaseFloat param_delta =
          std::sqrt(DotProduct(*delta_nnet_, *delta_nnet_)) * scale;
      if (param_delta > config_.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          ScaleNnet(0.0, delta_nnet_);
        } else {
          scale *= config_.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << config_.max_param_change
                    << ", scaling by " << config_.max_param_change / param_delta;
        }
      }
    }
    AddNnet(*delta_nnet_, scale, nnet_);
    // impose positivity for AffineComponent max(W,0)
    PositiveUpdatableWeights(nnet_);

    ScaleNnet(config_.momentum, delta_nnet_);
  }
  if (config_.write_cache != "") {
    Output ko(config_.write_cache,
      config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
  }
}

void NnetXvectorTrainer::ProcessOutputs(NnetComputer *computer) {
  for (int32 node_index = 0; node_index < nnet_->NumNodes(); node_index++) {
    if (nnet_->IsOutputNode(node_index)) {
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;
      // For each xvector output node, we expect two output nodes with name "s"
      // and "b", which store symmetric affine transformation and bias term
      // for xvector-objective computation.
      std::string xvector_name = nnet_->GetNodeName(node_index),
        s_name = "s", b_name = "b";
      if (nnet_->GetNodeIndex(s_name) == -1 || nnet_->GetNodeIndex(b_name) == -1)
        KALDI_ERR << "The nnet expected to have two output nodes with name s and b.";

      if (xvector_name == "output") {
        const CuMatrixBase<BaseFloat> &xvector_pairs = computer->GetOutput(xvector_name),
          &xvec_s = computer->GetOutput(s_name),
          &xvec_b = computer->GetOutput(b_name);
        CuMatrix<BaseFloat> xvector_deriv(xvector_pairs.NumRows(), xvector_pairs.NumCols(),
                                          kUndefined);
        int32 s_dim = xvector_pairs.NumCols() * (xvector_pairs.NumCols() + 1) / 2;

        // convert CuVector to CuSpMatrix
        CuSpMatrix<BaseFloat> xvec_s_sp(xvector_pairs.NumCols());
        xvec_s_sp.CopyFromVec(xvec_s.Row(0));

        CuVector<BaseFloat> deriv_s(s_dim);
        BaseFloat xvec_b_val = xvec_b(0,0), deriv_b;
        ComputeXvectorObjfAndDeriv(xvector_pairs, xvec_s_sp, xvec_b_val,
                                   (supply_deriv ? &xvector_deriv : NULL),
                                   (supply_deriv ? &deriv_s : NULL),
                                   (supply_deriv ? &deriv_b : NULL),
                                   NULL, // The raw scores aren't needed
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

        objf_info_[xvector_name].UpdateStats(xvector_name,
                                             config_.print_interval,
                                             num_minibatches_processed_++,
                                             tot_weight, tot_objf);
      }
    }
  }
}

bool NnetXvectorTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  std::vector<std::pair<std::string, const ObjectiveFunctionInfo*> > all_pairs;
  for (; iter != end; ++iter)
    all_pairs.push_back(std::pair<std::string, const ObjectiveFunctionInfo*>(
          iter->first, &(iter->second)));
  // ensure deterministic order of these names (this will matter in situations
  // where a script greps for the objective from the log).
  std::sort(all_pairs.begin(), all_pairs.end());
  bool ans = false;
  for (size_t i = 0; i < all_pairs.size(); i++) {
    const std::string &name = all_pairs[i].first;
    const ObjectiveFunctionInfo &info = *(all_pairs[i].second);
    bool ok = info.PrintTotalStats(name);
    ans = ans || ok;
  }
  return ans;
}

NnetXvectorTrainer::~NnetXvectorTrainer() {
  delete delta_nnet_;
}

void GetComputationRequestXvector(const Nnet &nnet,
                                  const NnetExample &eg,
                                  bool need_model_derivative,
                                  bool store_component_stats,
                                  ComputationRequest *request) {
  request->inputs.clear();
  request->inputs.reserve(eg.io.size());
  request->outputs.clear();
  request->outputs.reserve(eg.io.size());
  request->need_model_derivative = need_model_derivative;
  request->store_component_stats = store_component_stats;

  // Examples for xvectors have no outputs.
  for (size_t i = 0; i < eg.io.size(); i++) {
    const NnetIo &io = eg.io[i];
    const std::string &name = io.name;
    int32 node_index = nnet.GetNodeIndex(name);

    if (node_index == -1 &&
        !nnet.IsInputNode(node_index))
      KALDI_ERR << "xvector example has input  named '" << name
                << "', but no such input node is in the network.";

    std::vector<IoSpecification> &dest = request->inputs;
    dest.resize(dest.size() + 1);
    IoSpecification &io_spec = dest.back();
    io_spec.name = name;
    io_spec.indexes = io.indexes;
    io_spec.has_deriv = false;
  }

  // We only need the output on frame t=0 for each n.
  // So the output index for the output node is (n, 0, 0)
  // for n=0 to max(n).
  // Indexes for "s" and "b" output nodes are equal to (0,0,0).
  int32 io_index_size = request->inputs[0].indexes.size(),
        n_indx_size = 0;
  std::vector<Index> output_indexes,
    affine_output_indexes;
  affine_output_indexes.resize(1);
  affine_output_indexes[0].n = 0;
  affine_output_indexes[0].t = 0;

  for (int32 indx = 0; indx < io_index_size; indx++)
    n_indx_size = std::max(n_indx_size,
      request->inputs[0].indexes[indx].n + 1);

  output_indexes.resize(n_indx_size);
  for (int32 indx = 0; indx < n_indx_size; indx++) {
    output_indexes[indx].n = indx;
    output_indexes[indx].t = 0;
  }

  // In order to generate computation request for output nodes,
  // we should find output nodes and add io_spec for each one.
  int32 num_nodes = nnet.NumNodes();
  for (size_t node_index = 0; node_index < num_nodes; node_index++) {
    if (nnet.IsOutputNode(node_index)) {
      std::vector<IoSpecification> &dest = request->outputs;
      dest.resize(dest.size() + 1);
      IoSpecification &io_spec = dest.back();
      io_spec.name = nnet.GetNodeName(node_index);
      if (nnet.GetNodeName(node_index) == "s" ||
          nnet.GetNodeName(node_index) == "b")
        io_spec.indexes = affine_output_indexes;
      else
        io_spec.indexes = output_indexes;
      io_spec.has_deriv = need_model_derivative;
    }
  }

  // check to see if something went wrong.
  if (request->inputs.empty())
    KALDI_ERR << "No inputs in computation request.";
  if (request->outputs.empty())
    KALDI_ERR << "No outputs in computation request.";
}



} // namespace nnet3
} // namespace kaldi

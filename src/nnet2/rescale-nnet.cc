// nnet2/rescale-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/rescale-nnet.h"

namespace kaldi {
namespace nnet2 {


class NnetRescaler {
 public:
  NnetRescaler(const NnetRescaleConfig &config,
               const std::vector<NnetExample> &examples,
               Nnet *nnet):
      config_(config), examples_(examples), nnet_(nnet) {}
                            
  void Rescale();

 private:
  /// takes the input and formats as a single matrix, in forward_data_[0].
  void FormatInput(const std::vector<NnetExample> &data,
                   CuMatrix<BaseFloat> *input);
  void RescaleComponent(int32 c, int32 num_chunks,
                        CuMatrixBase<BaseFloat> *cur_data_in,
                        CuMatrix<BaseFloat> *next_data);

  void ComputeRelevantIndexes();
  
  BaseFloat GetTargetAvgDeriv(int32 c);
  
  const NnetRescaleConfig &config_;
  const std::vector<NnetExample> &examples_;
  Nnet *nnet_;
  std::set<int32> relevant_indexes_; // values of c with AffineComponent followed
  // by (at c+1) NonlinearComponent that is not SoftmaxComponent.
};


void NnetRescaler::FormatInput(const std::vector<NnetExample> &data,
                               CuMatrix<BaseFloat> *input) {
  KALDI_ASSERT(data.size() > 0);
  int32 num_splice = nnet_->LeftContext() + 1 + nnet_->RightContext();
  KALDI_ASSERT(data[0].input_frames.NumRows() == num_splice);

  int32 feat_dim = data[0].input_frames.NumCols(),
         spk_dim = data[0].spk_info.Dim(),
         tot_dim = feat_dim + spk_dim; // we append these at the neural net
                                       // input... note, spk_dim might be 0.
  KALDI_ASSERT(tot_dim == nnet_->InputDim());
  int32 num_chunks = data.size();

  input->Resize(num_splice * num_chunks,
                tot_dim);
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> dest(*input,
                                chunk * num_splice, num_splice,
                                0, feat_dim);
    Matrix<BaseFloat> src(data[chunk].input_frames);
    dest.CopyFromMat(src);
    if (spk_dim != 0) {
      CuSubMatrix<BaseFloat> spk_dest(*input,
                                      chunk * num_splice, num_splice,
                                      feat_dim, spk_dim);
      spk_dest.CopyRowsFromVec(data[chunk].spk_info);
    }
  }
}

void NnetRescaler::ComputeRelevantIndexes() {
  for (int32 c = 0; c + 1 < nnet_->NumComponents(); c++)
    if (dynamic_cast<AffineComponent*>(&nnet_->GetComponent(c)) != NULL &&
        (dynamic_cast<NonlinearComponent*>(&nnet_->GetComponent(c+1)) != NULL &&
         dynamic_cast<SoftmaxComponent*>(&nnet_->GetComponent(c+1)) == NULL))
      relevant_indexes_.insert(c);
}


BaseFloat NnetRescaler::GetTargetAvgDeriv(int32 c) {
  KALDI_ASSERT(relevant_indexes_.count(c) == 1);
  BaseFloat factor;
  if (dynamic_cast<SigmoidComponent*>(&(nnet_->GetComponent(c + 1))) != NULL)
    factor = 0.25;
  else if (dynamic_cast<TanhComponent*>(&(nnet_->GetComponent(c + 1))) != NULL)
    factor = 1.0;
  else
    KALDI_ERR << "This type of nonlinear component is not handled: index  " << c;
  
  int32 last_c = *std::max_element(relevant_indexes_.begin(), relevant_indexes_.end()),
      first_c = *std::min_element(relevant_indexes_.begin(), relevant_indexes_.end());
  if (c == first_c)
    return factor * config_.target_first_layer_avg_deriv;
  else if (c == last_c)
    return factor * config_.target_last_layer_avg_deriv;
  else
    return factor * config_.target_avg_deriv;
}

// Here, c is the index of the affine component, and
// c + 1 is the index of the nonlinear component; *cur_data is the
// output of the affine component.
void NnetRescaler::RescaleComponent(
    int32 c,
    int32 num_chunks,
    CuMatrixBase<BaseFloat> *cur_data_in,
    CuMatrix<BaseFloat> *next_data) {
  int32 rows = cur_data_in->NumRows(), cols = cur_data_in->NumCols();
  // Only handle sigmoid or tanh here.
  if (dynamic_cast<SigmoidComponent*>(&(nnet_->GetComponent(c + 1))) == NULL &&
      dynamic_cast<TanhComponent*>(&(nnet_->GetComponent(c + 1))) == NULL)
    KALDI_ERR << "This type of nonlinear component is not handled: index  " << c;
  // the nonlinear component:
  NonlinearComponent &nc =
      *(dynamic_cast<NonlinearComponent*>(&(nnet_->GetComponent(c + 1))));
  
  BaseFloat orig_avg_deriv, target_avg_deriv = GetTargetAvgDeriv(c);
  BaseFloat cur_scaling = 1.0; // current rescaling factor (on input).
  int32 num_iters = 10;
  
  CuMatrix<BaseFloat> cur_data(*cur_data_in),
      ones(rows, cols), in_deriv(rows, cols);
      
  ones.Set(1.0);
  nc.Propagate(cur_data, num_chunks, next_data);
  nc.Backprop(cur_data, *next_data, ones, num_chunks, NULL, &in_deriv);
  BaseFloat cur_avg_deriv;
  cur_avg_deriv = in_deriv.Sum() / (rows * cols);
  orig_avg_deriv = cur_avg_deriv;
  for (int32 iter = 0; iter < num_iters; iter++) {
    // We already have "cur_avg_deriv"; perturb the scale and compute
    // the next avg_deriv, so we can see how it changes with the scale.
    cur_data.CopyFromMat(*cur_data_in);
    cur_data.Scale(cur_scaling + config_.delta);
    nc.Propagate(cur_data, num_chunks, next_data);
    nc.Backprop(cur_data, *next_data, ones, num_chunks, NULL, &in_deriv);
    BaseFloat next_avg_deriv = in_deriv.Sum() / (rows * cols);
    KALDI_ASSERT(next_avg_deriv < cur_avg_deriv);
    // "gradient" is how avg_deriv changes as we change the scale.
    // should be negative.
    BaseFloat gradient = (next_avg_deriv - cur_avg_deriv) / config_.delta;
    KALDI_ASSERT(gradient < 0.0);
    BaseFloat proposed_change = (target_avg_deriv - cur_avg_deriv) / gradient;
    KALDI_VLOG(2) << "cur_avg_deriv = " << cur_avg_deriv << ", target_avg_deriv = "
                  << target_avg_deriv << ", gradient = " << gradient
                  << ", proposed_change " << proposed_change; 
    // Limit size of proposed change in "cur_scaling", to ensure stability.
    if (fabs(proposed_change / cur_scaling) > config_.max_change)
      proposed_change = cur_scaling * config_.max_change *
          (proposed_change > 0.0 ? 1.0 : -1.0);
    cur_scaling += proposed_change;
    
    cur_data.CopyFromMat(*cur_data_in);
    cur_data.Scale(cur_scaling);
    nc.Propagate(cur_data, num_chunks, next_data);
    nc.Backprop(cur_data, *next_data, ones, num_chunks, NULL, &in_deriv);
    cur_avg_deriv = in_deriv.Sum() / (rows * cols);
    if (fabs(proposed_change) < config_.min_change) break; // Terminate the
    // optimization
  }
  UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(
      &nnet_->GetComponent(c));
  KALDI_ASSERT(uc != NULL);
  uc->Scale(cur_scaling); // scale the parameters of the previous
  // AffineComponent.
  
  KALDI_LOG << "For component " << c << ", scaling parameters by "
            << cur_scaling << "; average "
            << "derivative changed from " << orig_avg_deriv << " to "
            << cur_avg_deriv << "; target was " << target_avg_deriv;
}
    


void NnetRescaler::Rescale() {
  ComputeRelevantIndexes(); // set up relevant_indexes_.
  CuMatrix<BaseFloat> cur_data, next_data;
  FormatInput(examples_, &cur_data);
  int32 num_chunks = examples_.size();
  for (int32 c = 0; c < nnet_->NumComponents(); c++) {
    Component &component = nnet_->GetComponent(c);
    if (relevant_indexes_.count(c - 1) == 1) {
      // the following function call also appropriately sets "next_data"
      // after doing the rescaling
      RescaleComponent(c - 1, num_chunks, &cur_data, &next_data);
    } else {
      component.Propagate(cur_data, num_chunks, &next_data);
    }
    cur_data.Swap(&next_data);
  }
}

void RescaleNnet(const NnetRescaleConfig &rescale_config,
                 const std::vector<NnetExample> &examples,
                 Nnet *nnet) {
  NnetRescaler rescaler(rescale_config, examples, nnet);
  rescaler.Rescale();
}


} // namespace nnet2
} // namespace kaldi

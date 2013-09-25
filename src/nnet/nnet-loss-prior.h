// nnet/nnet-loss-prior.h

// Copyright 2012  Karel Vesely

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

#ifndef KALDI_NNET_NNET_LOSS_PRIOR_H_
#define KALDI_NNET_NNET_LOSS_PRIOR_H_

#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace nnet1 {

class XentPrior {
 public:
  XentPrior() 
   : loss_(0.0), frames_(0), correct_(0),
     loss_nosil_(0.0), frames_nosil_(0), correct_nosil_(0),  
     loss_scaled_(0.0), frames_scaled_(0.0), correct_scaled_(0.0),  
     loss_scaled_nosil_(0.0), frames_scaled_nosil_(0.0), correct_scaled_nosil_(0.0),
     sil_pdfs_(-1)
  { }
  ~XentPrior() { }

  /// Evaluate cross entropy from soft labels
  void EvalVec(const CuMatrix<BaseFloat> &net_out, const std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff);

  /// Read the prior values
  void ReadPriors(std::string prior_rxfile, BaseFloat U = 1.0, BaseFloat S = 1.0, int32 num_S = 5);
  
  /// Generate string with error report
  std::string Report();

 private:
  //raw loss
  double loss_;
  int32 frames_;
  int32 correct_;
  //raw loss with sil excluded
  double loss_nosil_;
  int32 frames_nosil_;
  int32 correct_nosil_;
  //inv-prior scaled loss
  double loss_scaled_;
  double frames_scaled_;
  double correct_scaled_;
  //inv-prior scaled loss with sil excluded
  double loss_scaled_nosil_;
  double frames_scaled_nosil_;
  double correct_scaled_nosil_;

  CuArray<int32> max_id_;
  std::vector<int32> max_id_host_;

  CuArray<int32>  target_device_;
  CuVector<BaseFloat> log_post_tgt_;
  Vector<BaseFloat>   log_post_tgt_host_;

  /// Number of sil-model PDFs in the front
  int32 sil_pdfs_; 
  /// Inv-priors to rescale the errors
  Vector<BaseFloat>   inv_priors_;

};



} // namespace nnet1
} // namespace kaldi

#endif


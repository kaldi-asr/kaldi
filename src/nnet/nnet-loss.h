// nnet/nnet-loss.h

// Copyright 2011  Brno University of Technology (author: Karel Vesely)

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

#ifndef KALDI_NNET_NNET_LOSS_H_
#define KALDI_NNET_NNET_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {

class Xent {
 public:
  Xent() : frames_(0), correct_(0), loss_(0.0), entropy_(0.0), 
           frames_progress_(0), loss_progress_(0.0), entropy_progress_(0.0) { }
  ~Xent() { }

  /// Evaluate cross entropy from hard labels
  void Eval(const CuMatrixBase<BaseFloat> &net_out, const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff);
  /// Evaluate cross entropy from posteriors
  void Eval(const CuMatrixBase<BaseFloat> &net_out, const Posterior &target,
            CuMatrix<BaseFloat> *diff);
  /// Evaluate cross entropy from soft labels
  void EvalVec(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 frames_;
  int32 correct_;
  double loss_;
  double entropy_;

  // partial results during training
  int32 frames_progress_;
  double loss_progress_;
  double entropy_progress_;
  std::vector<float> loss_vec_;

  // loss computation buffers
  CuArray<int32>  target_device_;

  CuVector<BaseFloat> log_post_tgt_;
  Vector<BaseFloat>   log_post_tgt_host_;
  CuMatrix<BaseFloat> tgt_mat_device_;
  CuMatrix<BaseFloat> xentropy_aux_;

  // frame classification buffers 
  CuArray<int32> max_id_out_;
  std::vector<int32> max_id_out_host_;
  CuArray<int32> max_id_tgt_;
  std::vector<int32> max_id_tgt_host_;

};


class Mse {
 public:
  Mse() : frames_(0), loss_(0.0), 
          frames_progress_(0), loss_progress_(0) { }
  ~Mse() { }

  /// Evaluate mean square error from target values
  void Eval(const CuMatrixBase<BaseFloat>& net_out, const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff);
  void Eval(const CuMatrixBase<BaseFloat>& net_out, const Posterior& target,
            CuMatrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 frames_;
  double loss_;
  
  int32 frames_progress_;
  double loss_progress_;
  std::vector<float> loss_vec_;

  CuMatrix<BaseFloat> diff_pow_2_;
  CuVector<BaseFloat> sum_diff_pow_2_;
  Vector<BaseFloat>   sum_diff_pow_2_host_;
};



} // namespace nnet1
} // namespace kaldi

#endif


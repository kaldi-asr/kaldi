// nnet/nnet-loss.h

// Copyright 2011  Karel Vesely

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

#ifndef KALDI_NNET_LOSS_H
#define KALDI_NNET_LOSS_H

#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-stlvector.h"

namespace kaldi {

class Xent {
 public:
  Xent() : frames_(0), correct_(0), loss_(0.0) { }
  ~Xent() { }

  /// Evaluate cross entropy from hard labels
  void Eval(const CuMatrix<BaseFloat> &net_out, const CuMatrix<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff);
  /// Evaluate cross entropy from soft labels
  void EvalVec(const CuMatrix<BaseFloat> &net_out, const std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 frames_;
  int32 correct_;
  double loss_;
 
  CuStlVector<int32> max_id_;
  std::vector<int32> max_id_host_;

  CuStlVector<int32>  target_device_;
  CuVector<BaseFloat> log_post_tgt_;
  Vector<BaseFloat>   log_post_tgt_host_;

};



class Mse {
 public:
  Mse() : frames_(0), loss_(0.0) { }
  ~Mse() { }

  /// Evaluate mean square error from target values
  void Eval(const CuMatrix<BaseFloat> &net_out, const CuMatrix<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 frames_;
  double loss_;

  CuMatrix<BaseFloat> diff_pow_2_;
  CuVector<BaseFloat> sum_diff_pow_2_;
  Vector<BaseFloat>   sum_diff_pow_2_host_;

};



class MseProgress {
 public:
  MseProgress(int32 progress_step = 1e6) 
   : progress_step_(progress_step), progress_ctr_(0), 
     frames_(0), frames_progress_(0), 
     loss_(0.0), loss_progress_(0)
   { }
  ~MseProgress() { }

  /// Evaluate mean square error from target values
  void Eval(const CuMatrix<BaseFloat>& net_out, const CuMatrix<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 progress_step_;
  int32 progress_ctr_;

  int32 frames_;
  int32 frames_progress_;

  double loss_;
  double loss_progress_;

  std::vector<float> loss_vec_;

  CuMatrix<BaseFloat> diff_pow_2_;
  CuVector<BaseFloat> sum_diff_pow_2_;
  Vector<BaseFloat>   sum_diff_pow_2_host_;

};



} // namespace

#endif


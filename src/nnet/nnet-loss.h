// nnet/nnet-loss.h

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)

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

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


class LossItf {
 public:
  LossItf() { }
  virtual ~LossItf() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff) = 0;

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const Posterior &target,
            CuMatrix<BaseFloat> *diff) = 0;

  /// Generate string with error report,
  virtual std::string Report() = 0;

  /// Get loss value (frame average),
  virtual BaseFloat AvgLoss() = 0;
};


class Xent : public LossItf {
 public:
  Xent():
    frames_progress_(0.0),
    xentropy_progress_(0.0),
    entropy_progress_(0.0)
  { }

  ~Xent()
  { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff);

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const Posterior &target,
            CuMatrix<BaseFloat> *diff);

  /// Generate string with error report,
  std::string Report();

  /// Generate string with per-class error report,
  std::string ReportPerClass();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return (xentropy_.Sum() - entropy_.Sum()) / frames_.Sum();
  }

 private:
  // main stats collected per target-class,
  CuVector<double> frames_;
  Vector<double> correct_;
  CuVector<double> xentropy_;
  CuVector<double> entropy_;

  // partial results during training,
  double frames_progress_;
  double xentropy_progress_;
  double entropy_progress_;
  std::vector<float> loss_vec_;

  // weigting buffer,
  CuVector<BaseFloat> frame_weights_;
  CuVector<BaseFloat> target_sum_;

  // loss computation buffers,
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> frames_aux_;
  CuMatrix<BaseFloat> xentropy_aux_;
  CuMatrix<BaseFloat> entropy_aux_;

  // frame classification buffers,
  CuArray<int32> max_id_out_;
  CuArray<int32> max_id_tgt_;
};


class Mse : public LossItf {
 public:
  Mse():
    frames_(0.0),
    loss_(0.0),
    frames_progress_(0.0),
    loss_progress_(0.0)
  { }

  ~Mse()
  { }

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff);

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);

  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return loss_ / frames_;
  }

 private:
  double frames_;
  double loss_;

  double frames_progress_;
  double loss_progress_;
  std::vector<float> loss_vec_;

  CuVector<BaseFloat> frame_weights_;
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> diff_pow_2_;
};


class MultiTaskLoss : public LossItf {
 public:
  MultiTaskLoss()
  { }

  ~MultiTaskLoss() {
    while (loss_vec_.size() > 0) {
      delete loss_vec_.back();
      loss_vec_.pop_back();
    }
  }

  /// Initialize from string, the format for string 's' is :
  /// 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
  ///
  /// Practically it can look like this :
  /// 'multitask,xent,2456,1.0,mse,440,0.001'
  void InitFromString(const std::string& s);

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff) {
    KALDI_ERR << "This is not supposed to be called!";
  }

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);

  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss();

 private:
  std::vector<LossItf*>  loss_vec_;
  std::vector<int32>     loss_dim_;
  std::vector<BaseFloat> loss_weights_;

  std::vector<int32>     loss_dim_offset_;

  CuMatrix<BaseFloat>    tgt_mat_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_LOSS_H_


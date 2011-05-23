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
#include "matrix/matrix-lib.h"

namespace kaldi {

class Xent {
 public:
  Xent() : frames_(0), correct_(0), loss_(0.0) { }
  ~Xent() { }

  /// Evaluate cross entropy from hard labels
  void Eval(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target,
            Matrix<BaseFloat>* diff);
  /// Evaluate cross entropy from soft labels
  void Eval(const Matrix<BaseFloat>& net_out, const std::vector<int32>& target,
            Matrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 Correct(const Matrix<BaseFloat>& net_out,
                const std::vector<int32>& target);

 private:
  int32 frames_;
  int32 correct_;
  double loss_;
};

class Mse {
 public:
  Mse() : frames_(0), loss_(0.0) { }
  ~Mse() { }

  /// Evaluate mean square error from target values
  void Eval(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target,
            Matrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

 private:
  int32 frames_;
  double loss_;
};



} // namespace

#endif


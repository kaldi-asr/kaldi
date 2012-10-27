// nnet-cpu/am-nnet.h

// Copyright 2012  Johns Hopkins Universith (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_AM_NNET_H_
#define KALDI_NNET_CPU_AM_NNET_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet-cpu/nnet-nnet.h"

namespace kaldi {

/*
  The class AmNnet (AM stands for "acoustic model") has the job of taking the
  "Nnet1" class, which is a quite general neural network, and giving it an
  interface that's suitable for acoustic modeling; it deals with initializing it
  with 2-level trees, and with storing, and dividing by, the prior of each
  context-dependent state.
*/


class AmNnet {
 public:
  AmNnet() { }

  /// Initialize the neural network based acoustic model from a config file.
  /// At this point the priors won't be initialized; you'd have to do
  /// SetPriors for that.
  void Init(std::istream &config_is);
  
  int32 NumPdfs() const { return nnet_.OutputDim(); }

  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  const Nnet &GetNnet() const { return nnet_; }
  
  Nnet &GetNnet() { return nnet_; }

  void SetPriors(const VectorBase<BaseFloat> &priors);
  
  const VectorBase<BaseFloat> &Priors() const { return priors_; }

  std::string Info() const;

 private:
  Nnet nnet_;
  Vector<BaseFloat> priors_;
};



} // namespace

#endif // KALDI_NNET_CPU_AM_NNET_H_

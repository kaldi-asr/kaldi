// nnet2/am-nnet.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_AM_NNET_H_
#define KALDI_NNET2_AM_NNET_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet2/nnet-nnet.h"

namespace kaldi {
namespace nnet2 {

/*
  The class AmNnet (AM stands for "acoustic model") has the job of taking the
  "Nnet1" class, which is a quite general neural network, and giving it an
  interface that's suitable for acoustic modeling; it deals with storing, and
  dividing by, the prior of each context-dependent state.
*/


class AmNnet {
 public:
  AmNnet() { }

  AmNnet(const AmNnet &other): nnet_(other.nnet_), priors_(other.priors_) { }

  explicit AmNnet(const Nnet &nnet): nnet_(nnet) { }
  
  /// Initialize the neural network based acoustic model from a config file.
  /// At this point the priors won't be initialized; you'd have to do
  /// SetPriors for that.
  void Init(std::istream &config_is);

  /// Initialize from a neural network that's already been set up.
  /// Again, the priors will be empty at this point.
  void Init(const Nnet &nnet);

  int32 NumPdfs() const { return nnet_.OutputDim(); }
  
  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  const Nnet &GetNnet() const { return nnet_; }
  
  Nnet &GetNnet() { return nnet_; }

  void SetPriors(const VectorBase<BaseFloat> &priors);
  
  const VectorBase<BaseFloat> &Priors() const { return priors_; }

  std::string Info() const;

 private:
  const AmNnet &operator = (const AmNnet &other); // Disallow.
  Nnet nnet_;
  Vector<BaseFloat> priors_;
};



} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_AM_NNET_H_

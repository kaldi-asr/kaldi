// nnet3/am-nnet-simple.h

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_AM_NNET_H_
#define KALDI_NNET3_AM_NNET_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-nnet.h"

namespace kaldi {
namespace nnet3 {

/*
  The class AmNnetSimple (AM stands for "acoustic model") has the job of taking
  the "Nnet" class, which is a quite general neural network, and giving it an
  interface that's suitable for acoustic modeling, i.e. all the stuff that's
  specific to the speech recognition application, including dividing
  by the prior.

  We are calling it AmNnetSimple because it's intended to handle the restricted
  case where there is just one time-dependent type of input, called "input" in
  the network itself, plus possibly a non-time-dependent input called "ivector",
  and just one output called "output" (the posteriors).  We might later want
  other types of network that have possibly multiple different-named inputs, or
  might make use of the "x" dimension of the Index structure..  */


class AmNnetSimple {
 public:
  AmNnetSimple() { }

  AmNnetSimple(const AmNnetSimple &other):
      nnet_(other.nnet_),
      priors_(other.priors_),
      left_context_(other.left_context_),
      right_context_(other.right_context_) { }

  explicit AmNnetSimple(const Nnet &nnet): nnet_(nnet) { }
  
  /// Initialize from a neural network that's already been set up.
  /// The priors will be empty at this point.
  void Init(const Nnet &nnet);

  int32 NumPdfs() const;
  
  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

  const Nnet &GetNnet() const { return nnet_; }
  
  Nnet &GetNnet() { return nnet_; }

  void SetPriors(const VectorBase<BaseFloat> &priors);
  
  const VectorBase<BaseFloat> &Priors() const { return priors_; }

  std::string Info() const;

  /// Minimum left context required to compute an output.
  int32 LeftContext() const;

  /// Minimum right context required to compute an output.
  int32 RightContext() const;

  /// Returns the input feature dim.
  int32 InputDim() const;

  /// Returns the iVector dimension, or -1 if there is no such input.
  int32 IvectorDim() const;
  
 private:
  const AmNnetSimple &operator = (const AmNnetSimple &other); // Disallow.
  Nnet nnet_;
  Vector<BaseFloat> priors_;

  // The following variables are derived; they are re-computed
  // when we read the network or when it is changed.
  int32 left_context_;
  int32 right_context_;  
};



} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_AM_NNET_H_

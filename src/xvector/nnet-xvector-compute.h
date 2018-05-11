// xvector/nnet-xvector-compute.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)
//              2016  David Snyder
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

#ifndef KALDI_XVECTOR_NNET_XVECTOR_COMPUTE_H_
#define KALDI_XVECTOR_NNET_XVECTOR_COMPUTE_H_

#include "nnet3/nnet-am-decodable-simple.h" // For NnetSimpleComputationOptions
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "xvector/xvector.h"

namespace kaldi {
namespace nnet3 {

/**
  class NnetXvectorComputer is responsible for extracting xvectors from
  feature chunks.
**/
class NnetXvectorComputer {
 public:
  /// Constructor.
  NnetXvectorComputer(const NnetSimpleComputationOptions &opts,
                      Nnet *nnet);
  /// Extracts an xvector given input features.
  void ComputeXvector(const MatrixBase<BaseFloat> &feats,
                    Vector<BaseFloat> *xvector);
 private:
  Nnet *nnet_;
  const NnetSimpleComputationOptions config_;
  CachingOptimizingCompiler compiler_;

  /// Creates a computation request from the input features.
  void GetComputationRequest(const MatrixBase<BaseFloat> &feats,
                             ComputationRequest *request);
};
} // namespace nnet3
} // namespace kaldi

#endif //

// nnet3/xvector.

// Copyright 2017   Johns Hopkins University (author: Daniel Povey)
//           2017   Johns Hopkins University (author: Daniel Garcia-Romero)
//           2017   David Snyder
//           2018   Behavox Limited (author: Arseniy Gorin)

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

class XVectorExtractorParallelClass {

/*
This version is intended for multi-thread xvector extraction.
We allow compiler to be passed as a pointer.

IMPORTANT NOTE:

CachingOptimizingCompiler is not thread safe in terms of graph compilation.
To use this class without run-time errors with multiple threads,
one must make sure to pre-compile the graph cache in advance

*/

  public:
  XVectorExtractorParallelClass(
    const NnetSimpleComputationOptions &opts,
    const Nnet &nnet,
    CachingOptimizingCompiler *compiler,
    std::string utt,
    const int chunk_size,
    const int min_chunk_size,
    const Matrix<BaseFloat> &feats,
    BaseFloatVectorWriter *xvector_writer
  );

  void operator () (); 

  ~XVectorExtractorParallelClass ();

  private:
    void DeletePointers();
    KALDI_DISALLOW_COPY_AND_ASSIGN(XVectorExtractorParallelClass);

    static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
                                   const Nnet &nnet, CachingOptimizingCompiler *compiler,
                                   Vector<BaseFloat> *xvector);
    const NnetSimpleComputationOptions opts_;
    const Nnet *nnet_;
    CachingOptimizingCompiler &compiler_;
    std::string utt_;
    int chunk_size_;
    int min_chunk_size_;
    Matrix<BaseFloat> feats_;
    BaseFloatVectorWriter *xvector_writer_;

    BaseFloat tot_weight_;
    Vector<BaseFloat> xvector_avg_; // (nnet_->OutputDim("output"), kSetZero);
};

}}

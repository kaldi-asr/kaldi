// nnet2/nnet-example.h

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

#ifndef KALDI_NNET_CPU_NNET_EXAMPLE_H_
#define KALDI_NNET_CPU_NNET_EXAMPLE_H_

#include "nnet2/nnet-nnet.h"
#include "util/table-types.h"

namespace kaldi {


// NnetTrainingExample is the input data and corresponding labels (or labels)
// for one frame of input.  In the normal case there will be just one label,
// with a weight of 1.0.  But, for example, in discriminative training there
// might be a mixture of labels with different weights.
struct NnetTrainingExample {
  
  std::vector<std::pair<int32, BaseFloat> > labels;  
    
  Matrix<BaseFloat> input_frames; // The input data-- typically a number of frames
  // (nnet.LeftContext() + 1 + nnet.RightContext()) of raw features, not
  // necessarily contiguous.

  // The number of frames of left context (we can work out the #frames
  // of right context from input_frames.NumRows() and this).
  int32 left_context;
  
  Vector<BaseFloat> spk_info; // The speaker-specific input, if any;
  // a vector of possibly zero length.  We'll append this to each of the
  // input frames.
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};


typedef TableWriter<KaldiObjectHolder<NnetTrainingExample > > NnetTrainingExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetTrainingExample > > SequentialNnetTrainingExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetTrainingExample > > RandomAccessNnetTrainingExampleReader;



} // namespace

#endif // KALDI_NNET_CPU_NNET_EXAMPLE_H_

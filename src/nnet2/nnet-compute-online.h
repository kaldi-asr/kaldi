// nnet2/nnet-compute-online.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//                 Guoguo Chen

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

#ifndef KALDI_NNET2_NNET_COMPUTE_ONLINE_H_
#define KALDI_NNET2_NNET_COMPUTE_ONLINE_H_

#include "nnet2/nnet-nnet.h"

namespace kaldi {
namespace nnet2 {

/* This header provides functionality for doing forward computation in a situation
   where you want to start from the beginning of a file and progressively compute
   more, while re-using the hidden parts that (due to context) may be shared.
   (note: this sharing is more of an issue in multi-splice networks where there is
   splicing over time in the middle layers of the network).
   Note: this doesn't do the final taking-the-log and correcting for the prior.
   The current implementation is just an inefficient placeholder implementation;
   later we'll modify it to properly use previously computed activations.
*/

class NnetOnlineComputer {

 public:
  // All the inputs and outputs are of type CuMatrix, in case we're doing the
  // computation on the GPU (of course, if there is no GPU, it backs off to
  // using the CPU).
  // You should initialize an object of this type for each utterance you want
  // to decode.
  
  // Note: pad_input will normally be true; it means that at the start and end
  // of the file, we pad with repeats of the first/last frame, so that the total
  // number of frames it outputs is the same as the number of input frames.
  NnetOnlineComputer(const Nnet &nnet,
                     bool pad_input);

  // This function works as follows: given a chunk of input (interpreted
  // as following in time any previously supplied data), do the computation
  // and produce all the frames of output we can.  In the middle of the
  // file, the dimensions of input and output will be the same, but at
  // the beginning of the file, output will have fewer frames than input
  // due to required context.
  // It is the responsibility of the user to keep track of frame indices, if
  // required.  This class won't output any frame twice.
  void Compute(const CuMatrixBase<BaseFloat> &input,
               CuMatrix<BaseFloat> *output);
  
  // This flushes out the last frames of output; you call this when all
  // input has finished.  It's invalid to call Compute or Flush after
  // calling Flush.  It's valid to call Flush if no frames have been
  // input or if no frames have been output; this produces empty output.
  void Flush(CuMatrix<BaseFloat> *output);

 private:
  void Propagate();

  const Nnet &nnet_;

  // data_ contains the intermediate stages and the output of the most recent
  // computation.
  std::vector<CuMatrix<BaseFloat> > data_;

  std::vector<ChunkInfo> chunk_info_;

  CuMatrix<BaseFloat> unused_input_;

  bool pad_input_;

  bool is_first_chunk_;

  bool finished_;
  // we might need more variables here to keep track of how many frames we
  // already output from data_.

  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetOnlineComputer);
};


} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_COMPUTE_ONLINE_H_

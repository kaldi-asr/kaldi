// nnet2/nnet-compute.h

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

#ifndef KALDI_NNET2_NNET_COMPUTE_H_
#define KALDI_NNET2_NNET_COMPUTE_H_

#include "nnet2/nnet-nnet.h"

namespace kaldi {
namespace nnet2 {

/* This header provides functionality for doing forward computation and
   backpropagation for whole chunks of features, e.g. whole utterances.  The
   code in nnet-update.h is designed for sample-by-sample computation.
*/


/**
  Does the basic neural net computation, on a sequence of data (e.g.
  an utterance).  If pad_input==true we'll pad the input with enough
  frames of context, and the output will be a matrix of #frames by
  the output-dim of the network, typically representing state-level
  posteriors.   If pad_input==false we won't do this and the
  output will have a lower #frames than the input; we lose
  nnet.LeftContext() at the left and nnet.RightContext() at the
  output.
*/
void NnetComputation(const Nnet &nnet,
                     const CuMatrixBase<BaseFloat> &input,  // features
                     const CuVectorBase<BaseFloat> &spk_info,
                     bool pad_input,
                     CuMatrixBase<BaseFloat> *output); // posteriors.

/** Does the neural net computation and backprop, given input and labels.
    Note: if pad_input==true the number of rows of input should be the
    same as the number of labels, and if false, you should omit
    nnet.LeftContext() labels on the left and nnet.RightContext() on
    the right.  If nnet_to_update == &nnet, then this does stochastic
    gradient descent, otherwise (assuming you have called SetZero(true)
    on *nnet_to_update) it will compute the gradient on this data.
    Returns the total objective function summed over the frames, times
    the utterance weight.
*/
BaseFloat NnetGradientComputation(const Nnet &nnet,
                                  const MatrixBase<BaseFloat> &input,
                                  const VectorBase<BaseFloat> &spk_info,
                                  bool pad_input,
                                  BaseFloat utterance_weight,
                                  const std::vector<int32> &labels,
                                  Nnet *nnet_to_update);



} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_COMPUTE_H_

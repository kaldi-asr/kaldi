// chain/chain-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "chain/chain-training.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-denominator.h"

namespace kaldi {
namespace chain {

void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *tot_objf,
                              BaseFloat *tot_weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BaseFloat num_logprob_weighted;
  if (nnet_output_deriv)
    nnet_output_deriv->SetZero();
  {
    NumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, and the logprob too.
    num_logprob_weighted = numerator.Forward();
    if (nnet_output_deriv)
      numerator.Backward(nnet_output_deriv);
  }
  DenominatorComputation denominator(opts, den_graph,
                                     supervision.num_sequences,
                                     nnet_output);

  BaseFloat den_logprob = denominator.Forward();
  if (nnet_output_deriv)
    denominator.Backward(-supervision.weight,
                         nnet_output_deriv);

  *tot_objf = num_logprob_weighted - supervision.weight * den_logprob;
  *tot_weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;
  if (!(*tot_objf  == *tot_objf)) {
    // inf or NaN detected
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*tot_objf)
               << ", setting to " << default_objf << " per frame.";
    *tot_objf  = default_objf * *tot_weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }
}


}  // namespace chain
}  // namespace kaldi

// chain/chain-denominator.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_DENOMINATOR_H_
#define KALDI_CHAIN_CHAIN_DENOMINATOR_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-supervision.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace chain {

struct ChainTrainingOptions {
  BaseFloat pdf_boundary_penalty;

  ChainTrainingOptions():
      pdf_boundary_penalty(4.0);

  void Register(OptionsItf *opts) {
    opts->Register("pdf-boundary-penalty", &pdf_boundary_penalty,
                   "Value subtracted from pdf pseudo-likelihoods at the "
                   "first and last frames of chopped-up sequences, if "
                   "the pdf is not in the numerator.  Intended to help "
                   "reduce the inaccuracy of derivatives arising from breaking "
                   "up the training utterances into fixed-size pieces.  This "
                   "value should not be too large to avoid floating-point range "
                   "issues.");
};


// This does forward-backward in parallel on a number of sequences, using a
// single HMM.
class DenominatorComputation {
 public:
  /*
    Constructor.  'nnet_output' is the raw nnet output (which we'll treat as
    pseudo-log-likelihoods).

    @param [in] opts  The options.
    @param [in] graph  The HMM that we use for the denominator (like a decoding graph,
                       with pdf-ids on the transitions).
    @param [in] num_sequences The number of separate time sequences (all of the same length)
                       that we are working with.  Must divide nnet_output.NumRows().
    @param [in] nnet_output  The output of the neural network for this minibatch.
                       The rows must be ordered as (first frame of all sequences)
                       (second frame of all sequences), etc.
    @param [in] initial_pdf_ids Indexed by sequence, a list of the pdf-ids that
                      (in the numerator sequence) are active on the first frame.
                      Used to modify the nnet output, to limit sequences that
                      have that pdf-id as the initial frame.
    @param [in] final_pdf_ids Indexed by sequence, a list of the pdf-ids that
                     (in the numerator sequence) are active on the final frame.
                     Used to modify the nnet output, to limit sequences that
                     have that pdf-id as the final frame.
  */
  DenominatorComputation(const ChainTrainingOptions &opts,
                         const DenominatorGraph &graph,
                         int32 num_sequences,
                         const CuMatrixBase<BaseFloat> &nnet_output,
                         const std::vector<std::vector<int32 > > &initial_pdf_ids,
                         const std::vector<std::vector<int32 > > &final_pdf_ids);


  // Does the forward computation, and returns the total negated log-like summed
  // over all sequences.
  BaseFloat Forward();

  void Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);

 private:
  // sets up the alpha for frame t = 0.
  void AlphaFirstFrame();
  // the alpha computation for some 0 < t <= num_time_steps_.
  void AlphaGeneralFrame(int32 t);

  // done after all the alphas, this function computes and returns the total
  // log-likelihood summed over all the sequences, and sets tot_prob_ (if we're
  // doing correction) log_correction_term_.  Note, this won't be scaled by
  // 'deriv_scale' (which of course we haven't seen by the time this is called,
  // from the Forward() computation).
  BaseFloat ComputeTotLogLike();

  // backward computation without rearrangement.
  void BackwardInternal();

  void BetaLastFrame();
  // beta computation for 0 <= beta < num_time_steps_.
  void BetaGeneralFrame(int32 t);

  const ChainTrainingOptions &opts_;
  const DenominatorGraph &den_graph_;
  const std::vector<std::vector<int32 > > &initial_pdf_ids_;
  const std::vector<std::vector<int32 > > &final_pdf_ids_;

  const CuMatrixBase<BaseFloat> &nnet_output_;

  // number of separate frame sequences
  int32 num_sequences_;
  // number of frames per sequence.  nnet_output_.NumRows() equals
  // num_sequences_ * frames_per_sequence.
  int32 frames_per_sequence_;

  // The transpose of the nnet output (more convenient for memory locality).
  // Also, the first and last frames' pseudo-likelihooods are modified by
  // subtracting opts_.pdf_boundary_penalty for all pdfs not listed as being
  // initial/final in the supervision.
  CuMatrix<BaseFloat> nnet_output_transposed_;

  // the derivs w.r.t. the nnet outputs (transposed)
  CuMatrix<BaseFloat> nnet_output_deriv_tranposed_;

  // the alpha probabilities; dimension is (frames_per_sequence + 1) by (num-hmm-states
  // * num-sequences).  Note, they are not logs.
  CuMatrix<BaseFloat> alpha_;

  // the beta probabilities (rolling buffer); dimension is 2 * (num-hmm-states *
  // num-sequences).  Note: for efficiency and to simplify the equations, these
  // are actually the beta / tot_prob_.
  CuMatrix<BaseFloat> beta_;

  // the total probability for each sequence, excluding the product of
  // correction terms.  we multiply on each frame by 1/alpha of hmm-state 0 of
  // the previous frame; the products
  CuVector<BaseFloat> tot_prob_;

  // the log of tot_prob_.
  CuVector<BaseFloat> tot_log_prob_;

  // the log of the total correction term for each sequence, which is the
  // product of the alpha_[special hmm state] over all the frames.  The
  // 'correction terms' are terms that we divide the alphas and betas by in
  // order to keep them in a good dynamic range.  The product of them
  // must be included in the total likelihood.
  CuVector<BaseFloat> log_correction_term_;
};



}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_DENOMINATOR_H_


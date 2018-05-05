// chain/chain-generic-numerator.h

// Copyright       2017  Hossein Hadian
//                 2018 Johns Hopkins University (Jan "Yenda" Trmal)


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


#ifndef KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_
#define KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_

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
#include "chain/chain-datastruct.h"

namespace kaldi {
namespace chain {

/* This extended comment explains how end-to-end (i.e. flat-start) chain
   training is done and how it is mainly different from regular chain training.

   The key differnece with regular chain is that the end-to-end supervision FST
   (i.e. numerator graph) can have loops and more than one final state (we
   call it 'Generic' numerator in the code). This is because we do not
   have any alignments so we can't split the utterances and we can't remove
   the self-loops.
   Of course, the end-to-end FST still has to be epsilon-free and have pdf_id+1
   on its input and output labels, just like the regular supervision FST.
   The end-to-end supervision (which contains the generic numerator FST's) is
   created using TrainingGraphToSupervision from a training FST (i.e. an FST
   created using compile-train-graphs). It is stored in the same struct as
   regular supervision (i.e. chain::Supervision) but this function
   sets the 'e2e' flag to true. Also the generic  numerator FSTs
   are stored in 'e2e_fsts' instead of 'fst'.

   The TrainingGraphToSupervision function is called in nnet3-chain-e2e-get-egs
   binary to create end-to-end chain egs. The only difference between a regular
   and end-to-end chain example is the supervision as explained above.

   class GenericNumeratorComputation is responsible for doing Forward-Backward
   on a generic FST (i.e. the kind of FST we use in end-to-end chain
   training). It is the same as DenominatorComputation with 2 differences:
   [1] it runs on CPU
   [2] it does not use leakyHMM
   The F-B computation is done in log-domain.

   When the 'e2e' flag of a supervision is set, the ComputeChainObjfAndDeriv
   function in chain-training.cc uses GenericNumeratorComputation (instead
   of NumeratorCompuation) to compute the numerator derivatives.

   The implementation tries to optimize the memory transfers. The optimization
   uses the observation that for each supervision graph, only very limited
   number of pdfs is needed to evaluate the possible transitions from state
   to state. That means that for the F-B, we don't have to transfer the whole
   neural network output, we can copy only the limited set of pdfs activation
   values that will be needed for F-B on the given graph.

   To streamline things, in the constructor of this class, we remap the pdfs
   indices to a new space and store the bookkeeping info in the index_to_pdf_
   structure. This can be seen as if for each FST we create a subspace that
   has only the pdfs that are needed for the given FST (possibly ordered
   differently).

   Morover, we optimize memory transfers. The matrix of nnet outputs can be
   reshaped (viewed) as a matrix of dimensions
   (frames_per_sequence) x (num_sequences * pdf_stride), where the pdf_stride
   is the stride of the original matrix and pdf_stride >= num_pdfs.
   When the matrix is viewed this way, it becomes obvious that the pdfs of the
   k-th supervision sequence  have column index k * pdf_stride + original_pdf_index
   Once this is understood, the way how copy all pdfs in one shot should become
   obvious.

   The complete F-B is then done in this remapped space and only
   when copying the activation values from the GPU memory or copying
   the computed derivatives to GPU memory, we use the bookkeeping info to
   map the values correctly.
 */


// This class is responsible for the forward-backward of the
// end-to-end 'supervision' (numerator) FST. This kind of FST can
// have self-loops.
// Note: An end-to-end supervision is the same as a regular supervision
// (class chain::Supervision) except the 'e2e' flag is set to true
// and the numerator FSTs are stored in 'e2e_fsts' instead of 'fst'

class GenericNumeratorComputation {
 public:
  /// Initializes the object.
  GenericNumeratorComputation(const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output);

  // Does the forward-backward computation. Returns the total log-prob
  // multiplied by supervision_.weight.
  // In the backward computation, add (efficiently) the derivative of the
  // nnet output w.r.t. the (log-prob times supervision_.weight times
  // deriv_weight) to 'nnet_output_deriv'.
  bool ForwardBackward(BaseFloat *total_loglike,
                       CuMatrixBase<BaseFloat> *nnet_output_deriv);

  BaseFloat ComputeObjf();
 private:
  // For the remapped FSTs, copy the appropriate activations to CPU memory.
  // For explanation of what remapped FST is, see the large comment in the
  // beginning of the file
  void CopySpecificPdfsIndirect(
                             const CuMatrixBase<BaseFloat> &nnet_output,
                             const std::vector<MatrixIndexT> &indices,
                             Matrix<BaseFloat> *output);

  // For the remapped FSTs, copy the computed values back to gpu,
  // expand to the original shape and add to the output matrix.
  // For explanation of what remapped FST is, see the large comment in the
  // beginning of the file.
  void AddSpecificPdfsIndirect(
                             Matrix<BaseFloat> *logprobs,
                             const std::vector<MatrixIndexT> &indices,
                             CuMatrixBase<BaseFloat> *output);

  // sets up the alpha for frame t = 0.
  void AlphaFirstFrame(int seq, Matrix<BaseFloat> *alpha);

  // the alpha computation for 0 < t <= supervision_.frames_per_sequence
  // for some 0 <= seq < supervision_.num_sequences.
  BaseFloat AlphaRemainingFrames(int seq,
                              const Matrix<BaseFloat> &probs,
                              Matrix<BaseFloat> *alpha);

  // the beta computation for 0 <= t < supervision_.frames_per_sequence
  // for some 0 <= seq < supervision_.num_sequences.
  void BetaRemainingFrames(int32 seq,
                        const Matrix<BaseFloat> &probs,
                        const Matrix<BaseFloat> &alpha,
                        Matrix<BaseFloat> *beta,
                        Matrix<BaseFloat> *derivs);

  // the beta computation for t = supervision_.frames_per_sequence
  void BetaLastFrame(int seq,
                     const Matrix<BaseFloat> &alpha,
                     Matrix<BaseFloat> *beta);

  // returns total prob for the given matrix alpha (assumes the alpha
  // matrix was computed using AlphaFirstFrame() and AlphaRemainingFrames()
  // (it's exactly like 'tot_probe_' in DenominatorComputation)
  BaseFloat GetTotalProb(const Matrix<BaseFloat> &alpha);

  // some checking that we can do if debug mode is activated, or on frame zero.
  // Sets ok_ to false if a bad problem is detected.
  bool CheckValues(int32 seq,
                   const Matrix<BaseFloat> &probs,
                   const Matrix<BaseFloat> &alpha,
                   const Matrix<BaseFloat> &beta,
                   const Matrix<BaseFloat> &derivs) const;


  const Supervision &supervision_;

  // a reference to the nnet output.
  const CuMatrixBase<BaseFloat> &nnet_output_;
  int32 nnet_output_stride_;   // we keep the original stride extra
                               // as the matrix can change before ForwardBackward

  // in_transitions_ lists all the incoming transitions for
  // each state of each numerator graph
  // out_transitions_ does the same but for the outgoing transitions
  typedef std::vector<std::vector<DenominatorGraphTransition> > TransitionMap;
  std::vector<TransitionMap> in_transitions_, out_transitions_;
  std::vector<MatrixIndexT> index_to_pdf_;

  // final probs for each state of each numerator graph
  Matrix<BaseFloat> final_probs_;  // indexed by seq, state

  // an offset subtracted from the logprobs of transitions out of the first
  // state of each graph to help reduce numerical problems. Note the
  // generic forward-backward computations cannot be done in log-space.
  Vector<BaseFloat> offsets_;
};

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_

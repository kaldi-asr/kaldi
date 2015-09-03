// nnet2/nnet-update.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//           2014  Xiaohui Zhang

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

#ifndef KALDI_NNET2_NNET_UPDATE_H_
#define KALDI_NNET2_NNET_UPDATE_H_

#include "nnet2/nnet-nnet.h"
#include "nnet2/nnet-example.h"
#include "util/table-types.h"


namespace kaldi {
namespace nnet2 {

/** @file
   This header provides functionality for sample-by-sample stochastic
   gradient descent and gradient computation with a neural net.
   See also \ref nnet-compute.h which is the same thing but for
   whole utterances.
*/

class NnetEnsembleTrainer;

// This class NnetUpdater contains functions for updating the neural net or
// computing its gradient, given a set of NnetExamples. We
// define it in the header file becaused it's needed by the ensemble training.
// But in normal cases its functionality should be used by calling DoBackprop(),
// and by ComputeNnetObjf()
class NnetUpdater {
 public:
  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.  Note: nnet_to_update
  // may be NULL if you don't want do do backprop.
  NnetUpdater(const Nnet &nnet,
              Nnet *nnet_to_update);
  
  /// Does the entire forward and backward computation for this minbatch.
  /// Returns total objective function over this minibatch.  If tot_accuracy != NULL,
  /// outputs to that pointer the total accuracy.
  double ComputeForMinibatch(const std::vector<NnetExample> &data,
                             double *tot_accuracy);

  /// This version of ComputeForMinibatch is used when you have already called
  /// the function FormatNnetInput (defined below) to format your data as a
  /// single matrix.  This interface is provided because it can be more
  /// efficient to do this non-trivial CPU-based computation in a separate
  /// thread.  formatted_data is an input but this function will destroy it,
  /// which is why it's a pointer.
  double ComputeForMinibatch(const std::vector<NnetExample> &data,
                             Matrix<BaseFloat> *formatted_data,
                             double *tot_accuracy);
  
  void GetOutput(CuMatrix<BaseFloat> *output);
 protected:

  void Propagate();

  /// Formats the input as a single matrix and sets the size of forward_data_,
  /// and sets up chunk_info_out_.
  void FormatInput(const std::vector<NnetExample> &data);

  /// Computes objective function and derivative at output layer, but does not
  /// do the backprop [for that, see Backprop()].  Returns objf summed over all
  /// samples (with their weights).
  /// If tot_accuracy != NULL, it will output to tot_accuracy the sum over all labels
  /// of all examples, of (correctly classified ? 0 : 1) * weight-of-label.  This
  /// involves extra computation.
  double ComputeObjfAndDeriv(const std::vector<NnetExample> &data,
                             CuMatrix<BaseFloat> *deriv,
                             double *tot_accuracy = NULL) const;
  

  /// Backprop must be called after ComputeObjfAndDeriv.  Does the
  /// backpropagation; "nnet_to_update_" is updated.  Note: "deriv" will
  /// contain, at input, the derivative w.r.t. the output layer (as computed by
  /// ComputeObjfAndDeriv), but will be used as a temporary variable by this
  /// function.
  void Backprop(CuMatrix<BaseFloat> *deriv) const;

  friend class NnetEnsembleTrainer;
 private:
  // Must be called after Propagate().
  double ComputeTotAccuracy(const std::vector<NnetExample> &data) const;

  const Nnet &nnet_;
  Nnet *nnet_to_update_;
  int32 num_chunks_; // same as the minibatch size.
  std::vector<ChunkInfo> chunk_info_out_; 
  
  std::vector<CuMatrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of each of the components.

};


/// Takes the input to the nnet for a minibatch of examples, and formats as a
/// single matrix.  data.size() must be > 0.  Note: you will probably want to
/// copy this to CuMatrix after you call this function.
/// The num-rows of the output will, at exit, equal 
/// (1 + nnet.LeftContext() + nnet.RightContext()) * data.size().
/// The nnet is only needed so we can call LeftContext(), RightContext()
/// and InputDim() on it.
void FormatNnetInput(const Nnet &nnet,
                     const std::vector<NnetExample> &data,
                     Matrix<BaseFloat> *mat);


/// This function computes the objective function and either updates the model
/// or adds to parameter gradients.  Returns the cross-entropy objective
/// function summed over all samples (normalize this by dividing by
/// TotalNnetTrainingWeight(examples)).  It is mostly a wrapper for
/// a class NnetUpdater that's defined in nnet-update.cc, but we
/// don't want to expose that complexity at this level.
/// All these examples will be treated as one minibatch.
/// If tot_accuracy != NULL, it outputs to that pointer the total (weighted)
/// accuracy.
double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetExample> &examples,
                  Nnet *nnet_to_update,
                  double *tot_accuracy = NULL);

/// This version of DoBackprop allows you to separately call
/// FormatNnetInput and provide the result to DoBackprop; this
/// can be useful when using GPUs because the call to FormatNnetInput
/// can be in a separate thread from the one that uses the GPU.
/// "examples_formatted" is really an input, but it's a pointer
/// because internally we call Swap() on it, so we destroy
/// its contents.
double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetExample> &examples,
                  Matrix<BaseFloat> *examples_formatted,
                  Nnet *nnet_to_update,
                  double *tot_accuracy = NULL);



/// Returns the total weight summed over all the examples... just a simple
/// utility function.
BaseFloat TotalNnetTrainingWeight(const std::vector<NnetExample> &egs);

/// Computes objective function over a minibatch.  Returns the *total* weighted
/// objective function over the minibatch.
/// If tot_accuracy != NULL, it outputs to that pointer the total (weighted)
/// accuracy.
double ComputeNnetObjf(const Nnet &nnet,
                       const std::vector<NnetExample> &examples,
                       double *tot_accuracy= NULL);

/// This version of ComputeNnetObjf breaks up the examples into
/// multiple minibatches to do the computation.
/// Returns the *total* (weighted) objective function.
/// If tot_accuracy != NULL, it outputs to that pointer the total (weighted)
/// accuracy.
double ComputeNnetObjf(const Nnet &nnet,                          
                       const std::vector<NnetExample> &examples,
                       int32 minibatch_size,
                       double *tot_accuracy= NULL);


/// ComputeNnetGradient is mostly used to compute gradients on validation sets;
/// it divides the example into batches and calls DoBackprop() on each.
/// It returns the *average* objective function per frame.
double ComputeNnetGradient(
    const Nnet &nnet,
    const std::vector<NnetExample> &examples,
    int32 batch_size,
    Nnet *gradient);


} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_UPDATE_H_

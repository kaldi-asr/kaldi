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

/* This header provides functionality for sample-by-sample stochastic
   gradient descent and gradient computation with a neural net.
   See also nnet-compute.h which is the same thing but for
   whole utterances.
   This is the inner part of the training code; see nnet-train.h
   which contains a wrapper for this, with functionality for
   automatically keeping the learning rates for each layer updated
   using a heuristic involving validation-set gradients.
*/

// This class NnetUpdater contains functions for updating the neural net or
// computing its gradient, given a set of NnetExamples. We
// define it in the header file becaused it's needed by the ensemble training.
// But in normal cases its functionality should be used by calling DoBackprop(),
// and by ComputeNnetObjf()
class NnetEnsembleTrainer;
class NnetUpdater {
 public:
  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.  Note: nnet_to_update
  // may be NULL if you don't want do do backprop.
  NnetUpdater(const Nnet &nnet,
              Nnet *nnet_to_update);
  
  double ComputeForMinibatch(const std::vector<NnetExample> &data);
  // returns average objective function over this minibatch.
  
  void GetOutput(CuMatrix<BaseFloat> *output);
 protected:

  /// takes the input and formats as a single matrix, in forward_data_[0].
  void FormatInput(const std::vector<NnetExample> &data);
  
  // Possibly splices input together from forward_data_[component].
  //   MatrixBase<BaseFloat> &GetSplicedInput(int32 component, Matrix<BaseFloat> *temp_matrix);

  void Propagate();

  /// Computes objective function and derivative at output layer.
  double ComputeObjfAndDeriv(const std::vector<NnetExample> &data,
                             CuMatrix<BaseFloat> *deriv) const;
  
  /// Returns objf summed (and weighted) over samples.
  /// Note: "deriv" will contain, at input, the derivative w.r.t. the
  /// output layer but will be used as a temporary variable by
  /// this function.
  void Backprop(const std::vector<NnetExample> &data,
                CuMatrix<BaseFloat> *deriv);

  friend class NnetEnsembleTrainer;
 private:
  const Nnet &nnet_;
  Nnet *nnet_to_update_;
  int32 num_chunks_; // same as the minibatch size.
  
  std::vector<CuMatrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of each of the components.

  // These weights are one per parameter; they equal to the "weight"
  // member variables in the NnetExample structures.  These
  // will typically be about one on average.
  CuVector<BaseFloat> chunk_weights_;
};

/// This function computes the objective function and either updates the model
/// or adds to parameter gradients.  Returns the cross-entropy objective
/// function summed over all samples (normalize this by dividing by
/// TotalNnetTrainingWeight(examples)).  It is mostly a wrapper for
/// a class NnetUpdater that's defined in nnet-update.cc, but we
/// don't want to expose that complexity at this level.
/// All these examples will be treated as one minibatch.

double DoBackprop(const Nnet &nnet,
                  const std::vector<NnetExample> &examples,
                  Nnet *nnet_to_update);

/// Returns the total weight summed over all the examples... just a simple
/// utility function.
BaseFloat TotalNnetTrainingWeight(const std::vector<NnetExample> &egs);

/// Computes objective function over a minibatch.  Returns the *total* weighted
/// objective function over the minibatch.
double ComputeNnetObjf(const Nnet &nnet,
                       const std::vector<NnetExample> &examples);

/// This version of ComputeNnetObjf breaks up the examples into
/// multiple minibatches to do the computation.
/// Returns the *total* (weighted) objective function.
double ComputeNnetObjf(const Nnet &nnet,                          
                       const std::vector<NnetExample> &examples,
                       int32 minibatch_size);


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

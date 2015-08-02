// nnet2/combine-nnet-fast.h

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

#ifndef KALDI_NNET2_COMBINE_NNET_FAST_H_
#define KALDI_NNET2_COMBINE_NNET_FAST_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "util/parse-options.h"
#include "itf/options-itf.h"


// Compare with combine-nnet.h.  What we're doing is taking
// a set of neural nets, and combining them with combination weights
// (separate weights for each updatable layer), and optimizing
// these weights using a validation set,

// This is a faster implementation
// with multi-threading and more careful preconditioning.
// To get the pre-conditioning, we divide the validation subset
// up into small-ish batches (e.g. 100 frames), and compute the
// neural net gradient for each one.  We then compute the parameter
// gradient (i.e. the gradient w.r.t. the combination weights we're
// optimizing) for each batch, and use the scatter of these as a
// kind of Fisher matrix for preconditioning.

namespace kaldi {
namespace nnet2 {

/** Configuration class that controls neural net combination, where we combine a
    number of neural nets, trying to find for each layer the optimal weighted
    combination of the different neural-net parameters.
 */
struct NnetCombineFastConfig {
  int32 initial_model; // If provided, the index of the initial model to start
  // the optimization from.
  int32 num_lbfgs_iters; 
  int32 num_threads;
  BaseFloat initial_impr;
  BaseFloat fisher_floor; // Flooring value we use for Fisher matrix (mainly
                          // makes a difference in pnorm systems, where there
                          // are don't-care directions in parameter space.
  BaseFloat alpha; // A smoothing value we use in getting the Fisher matrix.
  int32 fisher_minibatch_size; // e.g. 64; a relatively small minibatch size we
  // use in the Fisher matrix computation (smaller will generally mean more accurate
  // preconditioning but will slow down the computation).
  int32 minibatch_size; // e.g. 1028; a larger minibatch size we use in
  // the gradient computation.
  int32 max_lbfgs_dim;
  BaseFloat regularizer;
  
  NnetCombineFastConfig(): initial_model(-1), num_lbfgs_iters(10),
                           num_threads(1), initial_impr(0.01), fisher_floor(1.0e-20),
                           alpha(0.01), fisher_minibatch_size(64), minibatch_size(1024),
                           max_lbfgs_dim(10), regularizer(0.0) {}
  
  void Register(OptionsItf *opts) {
    opts->Register("initial-model", &initial_model, "Specifies where to start the "
                   "optimization from.  If 0 ... #models-1, then specifies the model; "
                   "if >= #models, then the average of all inputs; if <0, chosen "
                   "automatically from the previous options.");
    opts->Register("num-lbfgs-iters", &num_lbfgs_iters, "Maximum number of function "
                   "evaluations for L-BFGS to use when optimizing combination weights");
    opts->Register("initial-impr", &initial_impr, "Amount of objective-function change "
                   "We aim for on the first iteration.");
    opts->Register("num-threads", &num_threads, "Number of threads to use in "
                   "multi-core computation");
    opts->Register("fisher-floor", &fisher_floor,
                   "Floor for diagonal of Fisher matrix (used in preconditioning)");
    opts->Register("alpha", &alpha, "Value we use in smoothing the Fisher matrix "
                   "with its diagonal, in preconditioning the update.");
    opts->Register("fisher-minibatch-size", &fisher_minibatch_size, "Size of minibatch "
                   "used in computation of Fisher matrix (smaller -> better "
                   "preconditioning");
    opts->Register("minibatch-size", &minibatch_size, "Minibatch size used in computing "
                   "gradients (only affects speed)");
    opts->Register("max-lbfgs-dim", &max_lbfgs_dim, "Maximum dimension to use in "
                   "L-BFGS (will not get higher than this even if the dimension "
                   "of the space gets higher.)");
    opts->Register("regularizer", &regularizer, "Add to the objective "
                   "function (which is average log-like per frame), -0.5 * "
                   "regularizer * square of parameters.");
  }  
};

void CombineNnetsFast(const NnetCombineFastConfig &combine_config,
                      const std::vector<NnetExample> &validation_set,
                      const std::vector<Nnet> &nnets_in,
                      Nnet *nnet_out);
  


} // namespace nnet2
} // namespace kaldi

#endif

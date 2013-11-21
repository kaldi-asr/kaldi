// nnet2/nnet-lbfgs.h

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

#ifndef KALDI_NNET2_NNET_LBFGS_H_
#define KALDI_NNET2_NNET_LBFGS_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {

// Note:the num-samples is determined by what you pipe in.
struct NnetLbfgsTrainerConfig {
  PreconditionConfig precondition_config;
  int32 minibatch_size;
  int32 lbfgs_dim; // Number of steps to keep in L-BFGS.
  int32 lbfgs_num_iters; // more precisely, the number of function evaluations.
  BaseFloat initial_impr;

  NnetLbfgsTrainerConfig(): minibatch_size(1024), lbfgs_dim(10),
                            lbfgs_num_iters(20), initial_impr(0.1) { }

  void Register(OptionsItf *po) {
    precondition_config.Register(po);
    po->Register("minibatch-size", &minibatch_size, "Size of minibatches used to "
                 "compute gradient information (only affects speed)");
    po->Register("lbfgs-dim", &lbfgs_dim, "Number of parameter/gradient vectors "
                 "to keep in L-BFGS (parameter \"m\" of L-BFGS).");
    po->Register("lbfgs-num-iters", &lbfgs_num_iters, "Number of function evaluations to do "
                 "in L-BFGS");
    po->Register("initial-impr", &initial_impr, "Improvement in objective "
                 "function per frame to aim for on initial iteration.");
  };
};

class NnetLbfgsTrainer {
 public:
  NnetLbfgsTrainer(const NnetLbfgsTrainerConfig &config): config_(config) { }

  void AddExample(const NnetExample &eg) { egs_.push_back(eg); }
  
  void Train(Nnet *nnet);

  ~NnetLbfgsTrainer();
 private:
  void Initialize(Nnet *nnet);

  void CopyParamsOrGradientFromNnet(const Nnet &nnet,
                                    VectorBase<BaseFloat> *params);
  void CopyParamsOrGradientToNnet(const VectorBase<BaseFloat> &params,
                                  Nnet *nnet);
  
  BaseFloat GetObjfAndGradient(const VectorBase<BaseFloat> &cur_value,
                               VectorBase<BaseFloat> *gradient);

  const Nnet *nnet_; // the original neural net.
  Nnet *nnet_precondition_; // This object stores the preconditioning
  // information, if do_precondition == true
  Vector<BaseFloat> params_; // Neural net parameters, stored as a vector.
  OptimizeLbfgs<BaseFloat> *lbfgs_;
  BaseFloat initial_objf_;
  const NnetLbfgsTrainerConfig &config_;
  std::vector<NnetExample> egs_;  
};


/** This function takes a neural net, and returns a neural net with the same
    structure, but with parameters set to zero (to represent a gradient), with
    all descendants of AffineComponent replaced with one of type
    AffineComponentA, which we can use (with the training examples) to compute
    preconditioning information.  */
Nnet *GetPreconditioner(const Nnet &nnet);

/** This function calls the function Precondition of class AffineComponentA in
    "preconditioner"; it multiplies the parameters of each descendant of
    AffineComponent in "nnet" in a way that's appropriate for converting from
    gradient-space to model-space, in methods like L-BFGS.  (This is like
    multiplying by an approximate inverse Hessian.)
    "preconditioner" is not const because of caching issues-- it may have to
    pre-compute certain quantities.  It is in a sense "really" const.
 */
void PreconditionNnet(const PreconditionConfig &config,
                      Nnet *preconditioner,
                      Nnet *nnet);


} // namespace nnet2
} // namespace kaldi

#endif

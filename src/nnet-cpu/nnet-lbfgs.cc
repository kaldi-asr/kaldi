// nnet/nnet-lbfgs.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-lbfgs.h"

namespace kaldi {


/// This function replaces any components of type
/// AffineComponent (or child), with components of type
/// AffineComponentA.  This type of component can store
/// the preconditioning information in the neural net.
void ReplaceAffineComponentsInNnet(Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    AffineComponent *ac = dynamic_cast<AffineComponent*>(
        &(nnet->GetComponent(c)));
    if (ac != NULL) {
      AffineComponentA *ac_new = new AffineComponentA(*ac);
      nnet->SetComponent(c, ac_new);
      ac_new->InitializeScatter();
    }
  }
}



void NnetLbfgsTrainer::Train(Nnet *nnet_in) {

  Nnet cur_nnet(*nnet_in);
  Nnet nnet_precondition(*nnet_in); // Create component that can do the
  // preconditioning.
  if (config_.precondition_config.do_precondition) {
    ReplaceAffineComponentsInNnet(&nnet_precondition);
  }
  bool is_gradient = true;
  nnet_precondition.SetZero(is_gradient);

  // This will store the preconditioning information in nnet_precondition
  // (if do_precondition == true), as well as the gradient.
  ComputeNnetGradient(cur_nnet, egs_, config_.minibatch_size,
                      &nnet_precondition);
  
}




} // namespace

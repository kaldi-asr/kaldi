// nnet2/widen-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/widen-nnet.h"
#include "gmm/model-common.h" // for GetSplitTargets()
#include <numeric> // for std::accumulate

namespace kaldi {
namespace nnet2 {


void AffineComponent::Widen(int32 new_dim,
                            BaseFloat param_stddev,
                            BaseFloat bias_stddev,
                            std::vector<NonlinearComponent*> c2, // will usually
                                                                 // have just
                                                                 // one element.
                            AffineComponent *c3) {
  int32 old_dim = this->OutputDim(), extra_dim = new_dim - old_dim;
  KALDI_ASSERT(!c2.empty());
  if (new_dim <= old_dim) {
    KALDI_WARN << "Not widening component because new dim "
               << new_dim << " <= old dim " << old_dim;
    return;
  }
  
  this->bias_params_.Resize(new_dim,
                            kCopyData);
  this->bias_params_.Range(old_dim, extra_dim).SetRandn();
  this->bias_params_.Range(old_dim, extra_dim).Scale(bias_stddev);

  this->linear_params_.Resize(new_dim, InputDim(), kCopyData);
  this->linear_params_.Range(old_dim, extra_dim,
                             0, InputDim()).SetRandn();
  this->linear_params_.Range(old_dim, extra_dim,
                             0, InputDim()).Scale(param_stddev);

  for (size_t i = 0; i < c2.size(); i++) // Change dimension of nonlinear
    c2[i]->SetDim(new_dim);              // components
    
  // Change dimension of next affine component [extend with zeros,
  // so the existing outputs do not change in value]
  c3->linear_params_.Resize(c3->OutputDim(), new_dim, kCopyData);
}

void WidenNnet(const NnetWidenConfig &widen_config,
               Nnet *nnet) {

  int32 C = nnet->NumComponents();
  int32 num_widened = 0;

  for (int32 c = 0; c < C - 3; c++) {
    AffineComponent *c1 = dynamic_cast<AffineComponent*>(&(nnet->GetComponent(c)));
    if (c1 == NULL) continue;
    std::vector<NonlinearComponent*> c2; // normally just one element, but allow two right now.
    c2.push_back(dynamic_cast<NonlinearComponent*>(&(nnet->GetComponent(c+1))));
    if (c2.back() == NULL) continue;
    c2.push_back(dynamic_cast<NonlinearComponent*>(&(nnet->GetComponent(c+2))));
    AffineComponent *c3;
    if (c2.back() == NULL) {
      c2.pop_back();
      c3 = dynamic_cast<AffineComponent*>(&(nnet->GetComponent(c+2)));
    } else {
      if (c + 3 >= C) continue;
      c3 = dynamic_cast<AffineComponent*>(&(nnet->GetComponent(c+3)));
    }
    if (c3 == NULL) continue;
    BaseFloat param_stddev = widen_config.param_stddev_factor /
        sqrt(1.0 * c1->InputDim());
    KALDI_LOG << "Widening component " << c << " from "
              << c1->OutputDim() << " to " << widen_config.hidden_layer_dim;
    
    c1->Widen(widen_config.hidden_layer_dim,
              param_stddev, widen_config.bias_stddev,
              c2, c3);
    num_widened++;
  }
  nnet->Check();
  KALDI_LOG << "Widened " << num_widened << " components.";
}  

  
} // namespace nnet2
} // namespace kaldi

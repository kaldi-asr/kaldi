// nnet2/nnet-fix.cc

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

#include "nnet2/nnet-fix.h"

namespace kaldi {
namespace nnet2 {


/* See the header for what we're doing.
   The pattern we're looking for is AffineComponent followed by
   a NonlinearComponent of type SigmoidComponent or TanhComponent.
*/

void FixNnet(const NnetFixConfig &config, Nnet *nnet) {
  for (int32 c = 0; c + 1 < nnet->NumComponents(); c++) {
    AffineComponent *ac = dynamic_cast<AffineComponent*>(
        &(nnet->GetComponent(c)));
    NonlinearComponent *nc = dynamic_cast<NonlinearComponent*>(
        &(nnet->GetComponent(c + 1)));
    if (ac == NULL || nc == NULL) continue;
    // We only want to process this if it's of type SigmoidComponent
    // or TanhComponent.
    BaseFloat max_deriv; // The maximum derivative of this nonlinearity.
    bool is_relu = false;
    {
      SigmoidComponent *sc = dynamic_cast<SigmoidComponent*>(nc);
      TanhComponent *tc = dynamic_cast<TanhComponent*>(nc);
      RectifiedLinearComponent *rc = dynamic_cast<RectifiedLinearComponent*>(nc);
      if (sc != NULL) max_deriv = 0.25;
      else if (tc != NULL) max_deriv = 1.0;
      else if (rc != NULL) { max_deriv = 1.0; is_relu = true; }
      else continue; // E.g. SoftmaxComponent; we don't handle this.
    }
    double count = nc->Count();
    Vector<double> deriv_sum (nc->DerivSum());
    if (count == 0.0 || deriv_sum.Dim() == 0) {
      KALDI_WARN << "Cannot fix neural net because no statistics are stored.";
      continue;
    }
    Vector<BaseFloat> bias_params(ac->BiasParams());
    Matrix<BaseFloat> linear_params(ac->LinearParams());
    int32 dim = nc->InputDim(), num_reduced = 0, num_increased = 0;
    for (int32 d = 0; d < dim; d++) {
      // deriv ratio is the ratio of the computed average derivative to the
      // maximum derivative of that nonlinear function.
      BaseFloat deriv_ratio = deriv_sum(d) / (count * max_deriv);
      KALDI_ASSERT(deriv_ratio >= 0.0 && deriv_ratio < 1.01); // Or there is an
                                                              // error in the
      // math.
      if (deriv_ratio < config.min_average_deriv) { // derivative is too small, meaning
        // we've gone off into the "flat part" of the sigmoid.
        if (is_relu) {
          bias_params(d) += config.relu_bias_change;
        } else {
          BaseFloat parameter_factor = std::min(config.min_average_deriv /
                                                deriv_ratio,
                                                config.parameter_factor);
          // we need to reduce the parameters, so multiply by 1/parameter factor.
          bias_params(d) *= 1.0 / parameter_factor;
          linear_params.Row(d).Scale(1.0 / parameter_factor);
        }
        num_reduced++;
      } else if (deriv_ratio > config.max_average_deriv && !is_relu) { // derivative is too large,
        // meaning we're only in the linear part of the sigmoid, in the middle.
        BaseFloat parameter_factor = std::min(deriv_ratio / config.max_average_deriv,
                                              config.parameter_factor);
        // we need to increase the factors, so multiply by parameter_factor.
        bias_params(d) *= parameter_factor;
        linear_params.Row(d).Scale(parameter_factor);
        num_increased++;
      }
    }
    if (is_relu) {
      KALDI_LOG << "For layer " << c << " (ReLU units), changed bias for "
                << num_reduced << " indexes, out of a total of " << dim;
    } else {
      KALDI_LOG << "For layer " << c << ", decreased parameters for "
                << num_reduced << " indexes, and increased them for "
                << num_increased << " out of a total of " << dim;
    }
    ac->SetParams(bias_params, linear_params);
  }
}
  
  
} // namespace nnet2
} // namespace kaldi

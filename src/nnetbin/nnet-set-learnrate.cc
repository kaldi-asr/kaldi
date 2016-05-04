// nnetbin/nnet-set-learnrate.cc

// Copyright 2016,  Brno University of Technology
//                  (author: Katerina Zmolikova, Karel Vesely)

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

#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-activation.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
      "Sets learning rate coefficient inside of 'nnet1' model\n"
      "Usage: nnet-set-learnrate --components=<csl> --coef=<float> <nnet-in> <nnet-out>\n"
      "e.g.: nnet-set-learnrate --components=1:3:5 --coef=0.5 --bias-coef=0.1 nnet-in nnet-out\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    std::string components_str = "";
    po.Register("components", &components_str,
        "Select components by 'csl' of 1..N values. Layout is the same as in "
        "'nnet-info' output, (example 1:3:5)");

    float coef = 1.0,
          weight_coef = 1.0,
          bias_coef = 1.0;

    po.Register("coef", &coef,
        "Learn-rate coefficient for both weight matrices and biases.");
    po.Register("weight-coef", &weight_coef,
        "Learn-rate coefficient for weight matrices "
        "(used as: coef * weight_coef).");
    po.Register("bias-coef", &bias_coef,
        "Learn-rate coefficient for bias (used as: coef * bias_coef).");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_in_filename = po.GetArg(1),
      nnet_out_filename = po.GetArg(2);

    Nnet nnet;
    nnet.Read(nnet_in_filename);

    // A vector which contains indices of components,
    // where we will set the 'learn-rate coefficients',
    std::vector<int32> components;
    if (components_str != "") {
      // components were selected by the option,
      kaldi::SplitStringToIntegers(components_str, ":", false, &components);
    } else {
      // otherwise select all the components (1..Ncomp),
      for (int32 i = 1; i <= nnet.NumComponents(); i++) {
        components.push_back(i);
      }
    }

    // Setting the learning rate coefficients,
    for (int32 i = 0; i < components.size(); i++) {
      if (nnet.GetComponent(components[i]-1).IsUpdatable()) {
        UpdatableComponent& comp =
          dynamic_cast<UpdatableComponent&>(nnet.GetComponent(components[i]-1));
        comp.SetLearnRateCoef(coef * weight_coef);  // weight matrices, etc.,
        comp.SetBiasLearnRateCoef(coef * bias_coef);  // biases,
      }
    }

    // Write the 'nnet1' network,
    nnet.Write(nnet_out_filename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


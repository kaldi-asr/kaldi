// nnet-cpubin/nnet-am-copy.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet-cpu/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a (cpu-based) neural net and its associated transition model,\n"
        "possibly changing the binary mode\n"
        "Also supports multiplying all the learning rates by a factor\n"
        "(the --learning-rate-factor option) and setting them all to a given\n"
        "value (the --learning-rate options)\n"
        "\n"
        "Usage:  nnet-am-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-copy --binary=false 1.mdl text.mdl\n";
    
    bool binary_write = true;
    BaseFloat learning_rate_factor = 1.0, learning_rate = -1;
    std::string learning_rates = "";
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learning-rate-factor", &learning_rate_factor,
                "Before copying, multiply all the learning rates in the "
                "model by this factor.");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of \"updatable\" layers"
                "are set to this value.");
    po.Register("learning-rates", &learning_rates,
                "If supplied (a colon-separated list of learning rates), sets "
                "the learning rates of \"updatable\" layers to these values.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(2);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    if (learning_rate_factor != 1.0)
      am_nnet.GetNnet().ScaleLearningRates(learning_rate_factor);

    if (learning_rate >= 0)
      am_nnet.GetNnet().SetLearningRates(learning_rate);

    if (learning_rates != "") {
      std::vector<BaseFloat> learning_rates_vec;
      if (!SplitStringToFloats(learning_rates, ":", false, &learning_rates_vec)
          || static_cast<int32>(learning_rates_vec.size()) !=
             am_nnet.GetNnet().NumUpdatableComponents()) {
        KALDI_ERR << "Expected --learning-rates option to be a "
                  << "colon-separated string with "
                  << am_nnet.GetNnet().NumUpdatableComponents()
                  << " elements, instead got \"" << learning_rates << '"';
      }
      SubVector<BaseFloat> learning_rates_vector(&(learning_rates_vec[0]),
                                                 learning_rates_vec.size());
      am_nnet.GetNnet().SetLearningRates(learning_rates_vector);
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Copied neural net from " << nnet_rxfilename
              << " to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

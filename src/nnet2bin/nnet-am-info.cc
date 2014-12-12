// nnet2bin/nnet-am-info.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/am-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print human-readable information about the neural network\n"
        "acoustic model to the standard output\n"
        "Usage:  nnet-am-info [options] <nnet-in>\n"
        "e.g.:\n"
        " nnet-am-info 1.nnet\n";
        
    ParseOptions po(usage);

    bool print_learning_rates = false;

    po.Register("print-learning-rates", &print_learning_rates,
                "If true, instead of printing the normal info, print a "
                "colon-separated list of the learning rates for each updatable "
                "layer, suitable to give to nnet-am-copy as the argument to"
                "--learning-rates.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    if (print_learning_rates) {
      Vector<BaseFloat> learning_rates(am_nnet.GetNnet().NumUpdatableComponents());
      am_nnet.GetNnet().GetLearningRates(&learning_rates);
      int32 nc = learning_rates.Dim();
      for (int32 i = 0; i < nc; i++)
        std::cout << learning_rates(i) << (i < nc - 1 ? ":" : "");
      std::cout << std::endl;
      KALDI_LOG << "Printed learning-rate info for " << nnet_rxfilename;
    } else {
      std::cout << am_nnet.Info();
      KALDI_LOG << "Printed info about " << nnet_rxfilename;
    }
    
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}





// nnet2bin/nnet-init.cc

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
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    // TODO: specify in the usage message where the example scripts are.
    const char *usage =
        "Initialize the neural network from a config file with a line for each\n"
        "component.  Note, this only outputs the neural net itself, not the associated\n"
        "information such as the transition-model; you'll probably want to pipe\n"
        "the output into something like am-nnet-init.\n"
        "\n"
        "Usage:  nnet-init [options] <config-in> <raw-nnet-out>\n"
        "e.g.:\n"
        " nnet-init tree topo nnet.config 1.nnet\n";
    
    bool binary_write = true;
    int32 srand_seed = 0;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("srand", &srand_seed, "Seed for random number generator");
    
    po.Read(argc, argv);
    srand(srand_seed);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string config_rxfilename = po.GetArg(1),
        raw_nnet_wxfilename = po.GetArg(2);
    
    Nnet nnet;
    {
      bool binary;
      Input ki(config_rxfilename, &binary);
      KALDI_ASSERT(!binary && "Expect config file to contain text.");
      nnet.Init(ki.Stream());
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Initialized raw neural net and wrote it to "
              << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

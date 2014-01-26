// nnetbin/nnet-initialize.cc

// Copyright 2014  Brno University of Technology (author: Karel Vesely)

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
#include "nnet/nnet-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy Neural Network model (and possibly change binary/text format)\n"
        "Usage:  nnet-initialize [options] <nnet-config-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-copy --binary=false nnet.conf nnet.init\n";

    SetVerboseLevel(1); // be verbose by default

    ParseOptions po(usage);
    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");
    int32 seed = 777;
    po.Register("seed", &seed, "Seed for random number generator");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_config_in_filename = po.GetArg(1),
        nnet_out_filename = po.GetArg(2);

    std::srand(seed);

    // initialize the network
    Nnet nnet;
    nnet.Init(nnet_config_in_filename); 
    
    // store the network
    Output ko(nnet_out_filename, binary_write);
    nnet.Write(ko.Stream(), binary_write);

    KALDI_LOG << "Written initialized model to " << nnet_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



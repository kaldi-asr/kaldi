// nnetbin/nnet-info.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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
        "Print human-readable information about the neural network.\n"
        "(topology, various weight statistics, etc.) It prints to stdout.\n"
        "Usage:  nnet-info [options] <nnet-in>\n"
        "e.g.:\n"
        " nnet-info 1.nnet\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1);

    // load the network
    Nnet nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    std::cout << nnet.Info();

    KALDI_LOG << "Printed info about " << nnet_rxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



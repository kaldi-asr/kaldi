// nnet2bin/raw-nnet-concat.cc

// Copyright 2013  Johns Hopkins University (author:  Daniel Povey)

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

    const char *usage =
        "Concatenate two 'raw' neural nets, e.g. as output by nnet-init or\n"
        "nnet-to-raw-nnet\n"
        "\n"
        "Usage:  raw-nnet-concat [options] <raw-nnet1-in> <raw-nnet2-in> <raw-nnet-out>\n"
        "e.g.:\n"
        " raw-nnet-concat nnet1 nnet2 nnet_concat\n";
    
    bool binary_write = true;
    int32 srand_seed = 0;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);
    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet1_rxfilename = po.GetArg(1),
        raw_nnet2_rxfilename = po.GetArg(2),
        raw_nnet_wxfilename = po.GetArg(3);
    
    Nnet nnet1;
    ReadKaldiObject(raw_nnet1_rxfilename, &nnet1);
    Nnet nnet2;
    ReadKaldiObject(raw_nnet2_rxfilename, &nnet2);

    Nnet nnet_concat(nnet1, nnet2); // Constructor concatenates them.

    WriteKaldiObject(nnet_concat, raw_nnet_wxfilename, binary_write);
    
    KALDI_LOG << "Concatenated neural nets from "
              << raw_nnet1_rxfilename << " and " << raw_nnet2_rxfilename
              << " and wrote to " << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

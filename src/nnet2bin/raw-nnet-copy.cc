// nnet2bin/raw-nnet-copy.cc

// Copyright 2014 Johns Hopkins University (author:  Daniel Povey)

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
        "Copy a raw neural net (this version works on raw nnet2 neural nets,\n"
        "without the transition model.  Supports the 'truncate' option.\n"
        "\n"
        "Usage:  raw-nnet-copy [options] <raw-nnet-in> <raw-nnet-out>\n"
        "e.g.:\n"
        " raw-nnet-copy --binary=false 1.mdl text.mdl\n"
        "See also: nnet-to-raw-nnet, nnet-am-copy\n";
    
    int32 truncate = -1;
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("truncate", &truncate, "If set, will truncate the neural net "
                "to this many components by removing the last components.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1),
        raw_nnet_wxfilename = po.GetArg(2);
    
    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    
    if (truncate >= 0)
      nnet.Resize(truncate);

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);

    KALDI_LOG << "Copied raw neural net from " << raw_nnet_rxfilename
              << " to " << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

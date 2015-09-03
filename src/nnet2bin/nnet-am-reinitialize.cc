// nnet2bin/nnet-am-reinitialize.cc

// Copyright 2014  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "This program can used when transferring a neural net from one language\n"
        "to another (or one tree to another).  It takes a neural net and a\n"
        "transition model from a different neural net, resizes the last layer\n"
        "to match the new transition model, zeroes it, and writes out the new,\n"
        "resized .mdl file.  If the original model had been 'mixed-up', the associated\n"
        "SumGroupComponent will be removed.\n"
        "\n"
        "Usage:  nnet-am-reinitialize [options] <nnet-in> <new-transition-model> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-reinitialize 1.mdl exp/tri6/final.mdl 2.mdl\n";

    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        transition_model_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel orig_trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      orig_trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    TransitionModel new_trans_model;
    ReadKaldiObject(transition_model_rxfilename, &new_trans_model);

    am_nnet.ResizeOutputLayer(new_trans_model.NumPdfs());
    
    {
      Output ko(nnet_wxfilename, binary_write);
      new_trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Resized neural net from " << nnet_rxfilename
              << " to " << am_nnet.NumPdfs()
              << " pdfs, and wrote to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

// nnet2bin/nnet-am-average.cc

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
#include "nnet2/nnet-randomize.h"
#include "nnet2/combine-nnet-a.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program average (or sums, if --sum=true) the parameters over a number of neural nets.\n"
        "\n"
        "Usage:  nnet-am-average [options] <model1> <model2> ... <modelN> <model-out>\n"
        "\n"
        "e.g.:\n"
        " nnet-am-average 1.1.nnet 1.2.nnet 1.3.nnet 2.nnet\n";
    
    bool binary_write = true;
    bool sum = false;
    
    ParseOptions po(usage);
    po.Register("sum", &sum, "If true, sums instead of averages.");
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string
        nnet1_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(po.NumArgs());
    
    TransitionModel trans_model;
    AmNnet am_nnet1;
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet1.Read(ki.Stream(), binary_read);
    }

    int32 num_inputs = po.NumArgs() - 1;
    BaseFloat scale = (sum ? 1.0 : 1.0 / num_inputs);

    am_nnet1.GetNnet().Scale(scale);
    
    for (int32 i = 2; i <= num_inputs; i++) {
      bool binary_read;
      Input ki(po.GetArg(i), &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      AmNnet am_nnet;
      am_nnet.Read(ki.Stream(), binary_read);
      am_nnet1.GetNnet().AddNnet(scale, am_nnet.GetNnet());
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet1.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Averaged parameters of " << num_inputs
              << " neural nets, and wrote to " << nnet_wxfilename;
    return 0; // it will throw an exception if there are any problems.
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


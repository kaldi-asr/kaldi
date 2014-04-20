// nnet2bin/nnet-replace-last-layers.cc

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
#include "nnet2/nnet-functions.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "This program is for adding new layers to a neural-network acoustic model.\n"
        "It removes the last --remove-layers layers, and adds the layers from the\n"
        "supplied raw-nnet.  The typical use is to remove the last two layers\n"
        "(the softmax, and the affine component before it), and add in replacements\n"
        "for them newly initialized by nnet-init.  This program is a more flexible\n"
        "way of adding layers than nnet-insert, but the inserted network needs to\n"
        "contain replacements for the removed layers.\n"
        "\n"
        "Usage:  nnet-replace-last-layers [options] <nnet-in> <raw-nnet-to-insert-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-replace-last-layers 1.nnet \"nnet-init hidden_layer.config -|\" 2.nnet\n";

    bool binary_write = true;
    int32 remove_layers = 2;

    ParseOptions po(usage);
    
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("remove-layers", &remove_layers, "Number of final layers "
                "to remove before adding input raw network.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        raw_nnet_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    Nnet src_nnet; // the one we'll insert.
    ReadKaldiObject(raw_nnet_rxfilename, &src_nnet);

    
    // This function is declared in nnet-functions.h
    ReplaceLastComponents(src_nnet,
                          remove_layers,
                          &(am_nnet.GetNnet()));
    KALDI_LOG << "Removed " << remove_layers << " components and added "
              << src_nnet.NumComponents();
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Write neural-net acoustic model to " <<  nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

// nnetbin/nnet-copy.cc

// Copyright 2012  Brno University of Technology (author: Karel Vesely)

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
        "Usage:  nnet-copy [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " nnet-copy --binary=false nnet.mdl nnet_txt.mdl\n";


    bool binary_write = true;
    int32 remove_first_layers = 0;
    int32 remove_last_layers = 0;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("remove-first-layers", &remove_first_layers, "Remove N first layers (Components) from the MLP");
    po.Register("remove-last-layers", &remove_last_layers, "Remove N last layers (Components) from the MLP");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // load the network
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    // optionally remove N first layers
    if(remove_first_layers > 0) {
      for(int32 i=0; i<remove_first_layers; i++) {
        nnet.RemoveComponent(0);
      }
    }
   
    // optionally remove N last layers
    if(remove_last_layers > 0) {
      for(int32 i=0; i<remove_last_layers; i++) {
        nnet.RemoveLastComponent();
      }
    }

    // store the network
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



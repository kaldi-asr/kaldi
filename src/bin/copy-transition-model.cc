// bin/copy-transition-model.cc

// Copyright 2009-2011 Microsoft Corporation

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

#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "util/common-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Copies a transition model (this can be used to separate transition \n"
        " models from the acoustic models they are written with.\n"
        "Usage:  copy-transition-model [options] <transition-model or model file> <transition-model-out>\n"
        "e.g.: \n"
        " copy-transition-model --binarhy=false 1.mdl 1.txt\n";

    bool binary;
    
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write output in binary mode.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transition_model_rxfilename = po.GetArg(1),
        transition_model_wxfilename = po.GetArg(2);


    TransitionModel trans_model;
    ReadKaldiObject(transition_model_rxfilename, &trans_model);

    WriteKaldiObject(trans_model, transition_model_wxfilename, binary);

    KALDI_LOG << "Copied transition model.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


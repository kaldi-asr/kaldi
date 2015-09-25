// ctcbin/nnet3-transition-model-info.cc

// Copyright 2015  Johns Hopkins University (author:  Daniel Povey)

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "ctc/cctc-transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print info about CTC transition model to the standard output\n"
        "\n"
        "Usage:  ctc-transition-model-info [options] <ctc-transition-model-in>\n"
        "e.g.:\n"
        " ctc-transition-model-info 0.trans_mdl\n";

    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string ctc_trans_model_rxfilename = po.GetArg(1);
    
    CctcTransitionModel trans_model;
    ReadKaldiObject(ctc_trans_model_rxfilename, &trans_model);
    
    WriteKaldiObject(trans_model, ctc_trans_model_wxfilename, binary_write);
    KALDI_LOG << "Copied CTC transition model from "
              << ctc_trans_model_rxfilename << " to "
              << ctc_trans_model_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

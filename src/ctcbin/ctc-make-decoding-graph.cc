// ctcbin/ctc-make-decoding-graph.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "fst/fstlib.h"
#include "ctc/cctc-graph.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;
    
    const char *usage =
        "Executes the last stages of creating a CTC decoding graph,\n"
        "given an LG.fst on the input, e.g. min(det(L o G))).\n"
        "\n"
        "Usage:  ctc-make-decoding-graph <ctc-transition-model> <in-fst> <out-fst>\n"
        "E.g:  ctc-make-decoding-graph final.mdl LG.fst > G.fst\n";
    

    ParseOptions po(usage);
    
    BaseFloat phone_lm_weight = 0.0;
    
    po.Register("phone-lm-weight", &phone_lm_weight,
                "The language-model weight to apply to the phone language "
                "model that the CCTC system was trained with... this would "
                "normally 0 [the theoretically best value], or positive but "
                "close to zero.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string cctc_trans_model_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetOptArg(2),
        fst_wxfilename = po.GetOptArg(3);


    ctc::CctcTransitionModel trans_model;
    ReadKaldiObject(cctc_trans_model_rxfilename, &trans_model);
    
    VectorFst<StdArc> *fst = ReadFstKaldi(fst_rxfilename);
    
    ctc::ShiftPhonesAndAddBlanks(fst);

    VectorFst<StdArc> decoding_fst;
    ctc::CreateCctcDecodingFst(trans_model, phone_lm_weight,
                               *fst, &decoding_fst);
    delete fst;
    WriteFstKaldi(decoding_fst, fst_wxfilename);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


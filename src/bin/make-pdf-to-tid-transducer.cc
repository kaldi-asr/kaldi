// bin/make-pdf-to-tid-transducer.cc
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
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Make transducer from pdfs to transition-ids\n"
        "Usage:   make-pdf-to-tid-transducer model-filename [fst-out]\n"
        "e.g.: \n"
        " make-pdf-to-tid-transducer 1.mdl > pdf2tid.fst\n";
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() <1 || po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string trans_model_filename = po.GetArg(1);
    std::string fst_out_filename = po.GetOptArg(2);

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_filename, &trans_model);

    fst::VectorFst<fst::StdArc> *fst = GetPdfToTransitionIdTransducer(trans_model);

#if _MSC_VER
    if (fst_out_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    if (!fst->Write(fst_out_filename))
      KALDI_ERR << "Error writing fst to "
                << (fst_out_filename == "" ? "standard output" : fst_out_filename);
    delete fst;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


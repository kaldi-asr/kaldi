// gmmbin/gmm-init-pdf-from-global.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy GMM based model and replace a pdf with a new GMM\n"
        "Usage:  gmm-init-pdf-from-global [options] <model-in> <pdf-id> <gmm-in> <model-out>\n"
        "e.g.:\n"
        " gmm-init-pdf-from-global 1.mdl new.gmm 1 1.new.mdl \n";


    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        gmm_in_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4);

    int32 pdf_id;
    if (!ConvertStringToInteger(po.GetArg(2), &pdf_id)) {
      KALDI_ERR << "Unable to convert argument 2 (" << po.GetArg(2) 
                << ") to integer";
    }

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    if (pdf_id >= am_gmm.NumPdfs() || pdf_id < 0) {
      KALDI_ERR << "pdf-id " << pdf_id << " is not in the "
                << "expected range [0-" << am_gmm.NumPdfs() - 1 << "]";
    }

    { 
      bool binary_read;
      Input ki(gmm_in_filename, &binary_read);
      DiagGmm gmm;
      gmm.Read(ki.Stream(), binary_read);
      am_gmm.GetPdf(pdf_id).CopyFromDiagGmm(gmm);
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



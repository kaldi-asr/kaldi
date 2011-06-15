// sgmmbin/sgmm-init.cc

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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

#include "util/common-utils.h"
#include "sgmm/am-sgmm.h"
#include "sgmm/fmllr-sgmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize an SGMM from a trained full-covariance UBM and a specified"
        " model topology.\n"
        "Usage: sgmm-comp-prexform [options] <sgmm-in> <occs-in> <sgmm-out>\n";

    bool binary = false;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string sgmm_in_filename = po.GetArg(1),
        occs_filename = po.GetArg(2),
        sgmm_out_filename = po.GetArg(3);

    kaldi::AmSgmm sgmm_in;
    kaldi::TransitionModel trans_model;
    {
      bool binary_read;
      kaldi::Input is(sgmm_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
      sgmm_in.Read(is.Stream(), binary_read);
    }

    kaldi::Vector<kaldi::BaseFloat> occs;
    {
      bool binary_read;
      kaldi::Input is(occs_filename, &binary_read);
      occs.Read(is.Stream(), binary_read);
    }

    kaldi::SgmmFmllrGlobalParams fmllr_globals;
    sgmm_in.ComputeFmllrPreXform(occs, &fmllr_globals.pre_xform_,
                                 &fmllr_globals.inv_xform_,
                                 &fmllr_globals.mean_scatter_);

    {
      kaldi::Output os(sgmm_out_filename, binary);
      trans_model.Write(os.Stream(), binary);
      sgmm_in.Write(os.Stream(), binary, kaldi::kSgmmWriteAll);
      fmllr_globals.Write(os.Stream(), binary);
    }

    KALDI_LOG << "Written model to " << sgmm_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



// sgmmbin/sgmm-est-fmllrbasis.cc

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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

#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "hmm/transition-model.h"
#include "sgmm/am-sgmm.h"
#include "sgmm/fmllr-sgmm.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for SGMM training.\n"
        "Usage: sgmm-est-fmllrbasis [options] <model-in> <model-out> "
        "<stats-in1> [stats-in2 ...]\n";

    bool binary = true;
    int32 num_bases = 50;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode.");
    po.Register("num-bases", &num_bases,
                "Number of fMLLR basis matrices to estimate.");
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    kaldi::AmSgmm am_sgmm;
    kaldi::TransitionModel trans_model;
    kaldi::SgmmFmllrGlobalParams fmllr_globals;
    {
      bool binary_read;
      kaldi::Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_sgmm.Read(ki.Stream(), binary_read);
      fmllr_globals.Read(ki.Stream(), binary_read);
    }

    kaldi::SpMatrix<double> fmllr_grad_scatter;
    int32 dim = am_sgmm.FeatureDim();
    fmllr_grad_scatter.Resize(dim * (dim + 1), kaldi::kSetZero);

    for (int i = 3, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      fmllr_grad_scatter.Read(ki.Stream(), binary_read,
                              true /* add read values */);
    }

    kaldi::EstimateSgmmFmllrSubspace(fmllr_grad_scatter, num_bases, dim,
                                     &fmllr_globals);

    // Write out the accs
    {
      kaldi::Output ko(model_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_sgmm.Write(ko.Stream(), binary, kaldi::kSgmmWriteAll);
      fmllr_globals.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



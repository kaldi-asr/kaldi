// gmmbin/gmm-est-hlda.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "transform/hlda.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Do HLDA update\n"
        "Usage:  gmm-est-hlda [options] <model-in> <full-hlda-mat-in> <model-out> <full-hlda-mat-out> <partial-hlda-mat-out> <stats-in1> <stats-in2> ... \n"
        "e.g.: gmm-est-hlda 1.mdl 1.hldafull 2.mdl 2.hldafull 2.hlda 1.0.hacc 1.1.hacc ... \n";

    bool binary = true;  // write in binary if true.

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() < 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        hldafull_in_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3),
        hldafull_out_filename = po.GetArg(4),
        hldapart_out_filename = po.GetArg(5);


    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    HldaAccsDiagGmm hlda_accs;
    for (int32 i = 6; i <= po.NumArgs(); i++) {
      std::string acc_filename = po.GetArg(i);
      bool binary_in, add = true;
      Input ki(acc_filename, &binary_in);
      hlda_accs.Read(ki.Stream(), binary_in, add);
    }

    Matrix<BaseFloat> hlda_mat_full;
    ReadKaldiObject(hldafull_in_filename, &hlda_mat_full);
    KALDI_ASSERT(hlda_mat_full.NumRows() == hlda_accs.FeatureDim()
                 && hlda_mat_full.NumCols() == hlda_accs.FeatureDim());

    Matrix<BaseFloat> hlda_mat_part(hlda_accs.ModelDim(),
                                    hlda_accs.FeatureDim());

    BaseFloat objf_impr, count;
    hlda_accs.Update(&am_gmm, &hlda_mat_full, &hlda_mat_part, &objf_impr, &count);

    KALDI_LOG << "Updated HLDA, total objf impr is " << (objf_impr/count)
              << " over " << count << " frames, logdet is "
              << hlda_mat_full.LogDet();

    WriteKaldiObject(hlda_mat_full, hldafull_out_filename, binary);
    WriteKaldiObject(hlda_mat_part, hldapart_out_filename, binary);
    {
      Output ko(model_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



// bin/est-mllt.cc
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
#include "transform/mllt.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Do MLLT update\n"
        "Usage:  est-mllt [options] <mllt-mat-out> <stats-in1> <stats-in2> ... \n"
        "e.g.: est-mllt 2.mat 1a.macc 1b.macc ... \n"
        "Note: use compose-transforms <mllt-mat-out> <prev-mllt-mat> to combine with previous\n"
        "  MLLT or LDA transform, if any, and\n"
        "  gmm-transform-means to apply <mllt-mat-out> to GMM means.\n";

    bool binary = true;  // write in binary if true.

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string mllt_out_filename = po.GetArg(1);

    MlltAccs mllt_accs;
    for (int32 i = 2; i <= po.NumArgs(); i++) {
      std::string acc_filename = po.GetArg(i);
      bool binary_in, add = true;
      Input ki(acc_filename, &binary_in);
      mllt_accs.Read(ki.Stream(), binary_in, add);
    }

    Matrix<BaseFloat> mat(mllt_accs.Dim(), mllt_accs.Dim());
    mat.SetUnit();
    BaseFloat objf_impr, count;
    mllt_accs.Update(&mat, &objf_impr, &count);

    KALDI_LOG << "Overall objective function improvement for MLLT is "
              << (objf_impr/count) << " over " << count << " frames, logdet is "
              << mat.LogDet();

    Output ko(mllt_out_filename, binary);
    mat.Write(ko.Stream(), binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



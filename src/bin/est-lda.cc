// bin/est-lda.cc

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
#include "transform/lda-estimate.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate LDA transform using stats obtained with acc-lda.\n"
        "Usage:  est-lda [options] <lda-matrix-out> <lda-acc-1> <lda-acc-2> ...\n";

    bool binary = true;
    std::string full_matrix_wxfilename;
    LdaEstimateOptions opts;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write matrix in binary mode.");
    po.Register("write-full-matrix", &full_matrix_wxfilename,
                "Write full LDA matrix to this location.");
    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    LdaEstimate lda;
    std::string lda_mat_wxfilename = po.GetArg(1);

    for (int32 i = 2; i <= po.NumArgs(); i++) {
      bool binary_in, add = true;
      Input ki(po.GetArg(i), &binary_in);
      lda.Read(ki.Stream(), binary_in, add);
    }

    Matrix<BaseFloat> lda_mat;
    Matrix<BaseFloat> full_lda_mat;
    lda.Estimate(opts, &lda_mat, &full_lda_mat);
    WriteKaldiObject(lda_mat, lda_mat_wxfilename, binary);
    if (full_matrix_wxfilename != "") {
      Output ko(full_matrix_wxfilename, binary);
      full_lda_mat.Write(ko.Stream(), binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



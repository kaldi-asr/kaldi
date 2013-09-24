// bin/sum-matrices.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Sum matrices, e.g. stats for fMPE training\n"
        "Usage:  sum-matrices [options] <mat-out> <mat-in1> <mat-in2> ...\n"
        "e.g.:\n"
        " sum-matrices mat 1.mat 2.mat 3.mat\n";

    ParseOptions po(usage);
    bool binary = true;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    Matrix<BaseFloat> mat;

    for (int32 i = 2; i <= po.NumArgs(); i++) {
      bool binary_in;
      Input ki(po.GetArg(i), &binary_in);
      mat.Read(ki.Stream(), binary_in, true); // true == add.
      // This will crash if dimensions do not match.
    }

    Output ko(po.GetArg(1), binary);
    mat.Write(ko.Stream(), binary);

    KALDI_LOG << "Summed " << (po.NumArgs()-1) << " matrices "
              << " of dimension " << mat.NumRows() << " by " << mat.NumCols();
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



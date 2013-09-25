// featbin/splice-feats.cc

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
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Splice features with left and right context (e.g. prior to LDA)\n"
        "Usage: splice-feats [options] in-rspecifier out-wspecifier\n";

    ParseOptions po(usage);
    int32 left_context = 4, right_context = 4;


    po.Register("left-context", &left_context, "Number of frames of left context");
    po.Register("right-context", &right_context, "Number of frames of right context");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {
      Matrix<BaseFloat> spliced;
      SpliceFrames(kaldi_reader.Value(),
                   left_context,
                   right_context,
                   &spliced);
      kaldi_writer.Write(kaldi_reader.Key(), spliced);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



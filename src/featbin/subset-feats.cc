// featbin/subset-feats.cc

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy a subset of features (the first n feature files)\n"
        "Usually used where only a small amount of data is needed\n"
        "Note: if you want a specific subset, it's usually best to\n"
        "filter the original .scp file with utils/filter_scp.pl\n"
        "(possibly with the --exclude option)\n"
        "Usage: subset-feats [options] in-rspecifier out-wspecifier\n"
        "See also extract-rows, select-feats, subsample-feats\n";
    
    ParseOptions po(usage);
    
    int32 n = 10;
    po.Register("n", &n, "If nonnegative, copy the first n feature files.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(n >= 0);
    
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    int32 k = 0;
    for (; !kaldi_reader.Done() && k < n; kaldi_reader.Next(), k++)
      kaldi_writer.Write(kaldi_reader.Key(), kaldi_reader.Value());

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



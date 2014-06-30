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
#include <climits>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy a subset of features (the first n feature files)\n"
        "Usually used where only a small amount of data is needed\n"
        "Usage: subset-feats [options] in-rspecifier out-wspecifier\n"
        "See also extract-rows, select-feats, subsample-feats\n";

    ParseOptions po(usage);
    
    int32 n = 10;
    std::string include_rspecifier = "";
    bool include = false;
    std::string exclude_rspecifier = "";
    bool exclude = false;
    po.Register("n", &n, "If nonnegative, copy the first n feature files.");
    po.Register("include", &include_rspecifier,
                "only output features whose file names are included in the utt2spk list, --n is disabled");
    po.Register("exclude", &exclude_rspecifier, 
                 "only ouput features whose file names are excluded from the utt2spk list, --n is disabled");

    po.Read(argc, argv);
 
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    if (!include_rspecifier.empty()) {
      include = true; 
      n = INT_MAX;   
    }
    if (! exclude_rspecifier.empty()) {
      exclude = true; 
      n = INT_MAX;
    }
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(n >= 0);
    
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

    RandomAccessTokenVectorReader include_reader, exclude_reader;
    if (include) {
      include_reader.Open(include_rspecifier);
    }
    if (exclude) {
      exclude_reader.Open(exclude_rspecifier);
    }
    int32 k = 0;
    for (; !kaldi_reader.Done() && k < n; kaldi_reader.Next(), k++) {
      std::string utt = kaldi_reader.Key();
      if (include) {
        if (!include_reader.HasKey(utt))
          continue;
      }
      if (exclude) {
        if (exclude_reader.HasKey(utt))
          continue;
      }
      kaldi_writer.Write(utt, kaldi_reader.Value());
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



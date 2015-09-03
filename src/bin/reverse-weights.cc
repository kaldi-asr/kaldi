// bin/reverse-weights.cc

// Copyright 2009-2011 Chao Weng  Microsoft Corporation

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
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "Modify per-frame weights by outputting 1.0-weight (if --reverse=true);\n"
        "if --reverse=false, do nothing to them.\n"
        "Usage: reverse-weights weights-rspecifier weights-wspecifier\n";
    
    bool reverse = true;
    ParseOptions po(usage);
    po.Register("reverse", &reverse,
                "If true, reverse weights by setting to 1.0-weight; else do nothing.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string weights_rspecifier = po.GetArg(1),
        weights_wspecifier = po.GetArg(2);

    kaldi::SequentialBaseFloatVectorReader weights_reader(weights_rspecifier);
    kaldi::BaseFloatVectorWriter weights_writer(weights_wspecifier); 
    
    int32 num_done = 0;
    
    for (; !weights_reader.Done(); weights_reader.Next()) {
      std::string key = weights_reader.Key();
      Vector<BaseFloat> weights = weights_reader.Value();
      if (reverse) { // set each weight to 1.0-weight.
        weights.Scale(-1.0);
        weights.Add(1.0);
      }
      weights_writer.Write(key, weights);
      num_done++;
    }
    if (reverse) KALDI_LOG << "Done reversing " << num_done << " weights.";
    else KALDI_LOG << "Done copying " << num_done << " weights.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


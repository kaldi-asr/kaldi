// bin/dot-weights.cc

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
        "Takes two archives of vectors (typically representing per-frame weights)\n"
        "and for each utterance, outputs the dot product.\n"
        "Useful for evaluating the accuracy of silence classifiers.\n"
        "Usage: dot-weights weights-rspecifier1 weights-rspecifier2 float-wspecifier\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string weights1_rspecifier = po.GetArg(1),
        weights2_rspecifier = po.GetArg(2),
        float_wspecifier = po.GetArg(3);

    kaldi::SequentialBaseFloatVectorReader weights1_reader(weights1_rspecifier);
    kaldi::RandomAccessBaseFloatVectorReader weights2_reader(weights2_rspecifier);
    kaldi::BaseFloatWriter float_writer(float_wspecifier); 
    
    int32 num_done = 0, num_err = 0;
    
    for (; !weights1_reader.Done(); weights1_reader.Next()) {
      std::string key = weights1_reader.Key();
      const Vector<BaseFloat> &weights1 = weights1_reader.Value();
      if (!weights2_reader.HasKey(key)) {
        KALDI_WARN << "No weights for utterance " << key << " in second table.";
        num_err++;
      } else {
        const Vector<BaseFloat> &weights2 = weights2_reader.Value(key);
        // Next line will crash if different sizes.  This is the
        // behavior we want [for now].
        if (weights1.Dim() != weights2.Dim()) {
          KALDI_WARN << "Dimension mismatch for utterance " << key
                     << " : " << weights1.Dim() << " vs. " << weights2.Dim();
          num_err++;
          continue;
        }
        BaseFloat dot = VecVec(weights1, weights2);
        float_writer.Write(key, dot);
        num_done++;
      }
    }
    KALDI_LOG << "Done computing dot products of " << num_done
              << " weights; errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


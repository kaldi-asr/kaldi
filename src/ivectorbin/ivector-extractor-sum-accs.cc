// ivectorbin/ivector-extractor-sum-accs.cc

// Copyright 2013  Daniel Povey

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
#include "ivector/ivector-extractor.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    
    const char *usage =
        "Sum accumulators for training of iVector extractor\n"
        "Usage: ivector-extractor-sum-accs [options] <stats-in1> "
        "<stats-in2> ... <stats-inN> <stats-out>\n";

    bool binary = true;
    bool parallel = false;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("parallel", &parallel, "If true, the program makes sure to "
                "open all filehandles before reading for any (useful when "
                "summing accs from long processes)");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_wxfilename = po.GetArg(po.NumArgs());

    IvectorExtractorStats stats;

    if (parallel) {
      std::vector<kaldi::Input*> inputs(po.NumArgs() - 1);
      for (int i = 1; i < po.NumArgs(); i++) {
        std::string stats_in_filename = po.GetArg(i);
        inputs[i-1] = new kaldi::Input(stats_in_filename); // Don't try
        // to work out binary status yet; this would cause us to wait
        // for the output of that process.  We delay it till later.
      }
      for (size_t i = 1; i < po.NumArgs(); i++) {
        bool b;
        kaldi::InitKaldiInputStream(inputs[i-1]->Stream(), &b);
        bool add = true;
        stats.Read(inputs[i-1]->Stream(), b, add);
        delete inputs[i-1];
      }
    } else {
      for (int32 i = 1; i < po.NumArgs(); i++) {
        std::string stats_rxfilename = po.GetArg(i);
        KALDI_LOG << "Reading stats from " << stats_rxfilename;
        bool binary_in;
        Input ki(stats_rxfilename, &binary_in);
        bool add = true;
        stats.Read(ki.Stream(), binary_in, add);
      }
    }    
    WriteKaldiObject(stats, stats_wxfilename, binary);
    
    KALDI_LOG << "Wrote summed stats to " << stats_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



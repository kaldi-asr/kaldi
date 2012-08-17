// bin/vec-i32-to-counts.cc

// Copyright 2012 Karel Vesely (Brno University of Technology)

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

/** @brief Sums the pdf vectors to counts, this is used to obtain prior counts for hybrid decoding.
*/
#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Creates the counts from the int32 vectors (alignments).\n"
        "Usage:  pdf-to-counts  [options] <alignments-rspecifier> <counts-wxfilname>\n"
        "e.g.: \n"
        " pdf-to-counts ark:1.ali prior.counts\n";
    ParseOptions po(usage);
    
    bool binary = false;
    po.Register("binary", &binary, "write in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string alignments_rspecifier = po.GetArg(1),
        wxfilename = po.GetArg(2);

    SequentialInt32VectorReader reader(alignments_rspecifier);

    std::vector<int32> counts;
    int32 num_done = 0;
    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      std::vector<int32> alignment = reader.Value();

      for (size_t i = 0; i < alignment.size(); i++) {
        int32 value = alignment[i];
        if(value >= counts.size()) {
          counts.resize(value+1);
        }
        counts[value]++; // accumulate counts
      }

      num_done++;
    }

    //convert to BaseFloat and writ
    Vector<BaseFloat> counts_f(counts.size());
    for(int32 i=0; i<counts.size(); i++) {
      counts_f(i) = counts[i];
    }
    Output ko(wxfilename, binary);
    counts_f.Write(ko.Stream(),binary);

    KALDI_LOG << "Summed " << num_done << " int32 vectors to counts.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



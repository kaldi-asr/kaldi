// bin/analyze-counts.cc

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

#include <iomanip>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Creates the counts from the int32 vectors (alignments).\n"
        "Usage:  analyze-counts  [options] <alignments-rspecifier> <counts-wxfilname>\n"
        "e.g.: \n"
        " analyze-counts ark:1.ali prior.counts\n";
    ParseOptions po(usage);
    
    bool binary = false;
    po.Register("binary", &binary, "write in binary mode");

    bool rescale_to_probs = false;
    po.Register("rescale-to-probs", &rescale_to_probs, "rescale the output to probablities instead");

    bool show_histogram = false;
    po.Register("show-histogram", &show_histogram, "show histgram to standard output");

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

    //need at least one occurence for each tgt, so there is no nan during decoding
    for(size_t i = 0; i < counts.size(); i++) {
      if(counts[i] == 0) counts[i]++;
    }

    //convert to BaseFloat and write
    Vector<BaseFloat> counts_f(counts.size());
    for(int32 i=0; i<counts.size(); i++) {
      counts_f(i) = counts[i];
    }
    //optionally rescale to probs
    if(rescale_to_probs) {
      counts_f.Scale(1.0/counts_f.Sum());
    }
    Output ko(wxfilename, binary);
    counts_f.Write(ko.Stream(),binary);
    //optionally show histogram
    if(show_histogram) {
      int32 n_bins=20;
      BaseFloat min = counts_f.Min(), max = counts_f.Max();
      BaseFloat step = (max-min)/n_bins;
      //accumulate bins
      int32 zero_bin = 0;
      std::vector<int32> hist_bin(n_bins+1);
      for (int32 i=0; i<counts_f.Dim(); i++) {
        if(counts_f(i) == 0.0) { 
          zero_bin++;
        } else {
          hist_bin[floor((counts_f(i)-min)/step)]++;
        }
      }
      //print the histogram
      using namespace std;
      int32 w=6, w2=3;
      std::cerr << "\%\%\% Histogram of the vector elements \%\%\%\n";
      std::cerr << "min : " << min << "  max: " << max << "\n";
      std::cerr << setw(w) << zero_bin << "\t" << setprecision(w2) << 0.0 << " exactly\n";
      for (int32 i=0; i<hist_bin.size(); i++) {
        std::cerr << setw(w) << hist_bin[i] << "\t" << setprecision(w2) << min+i*step << " to " << setprecision(w2) <<  min+(i+1)*step << "\n";
      }
      std::cerr << "\%\%\%\n";
    }

    KALDI_LOG << "Summed " << num_done << " int32 vectors to counts.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



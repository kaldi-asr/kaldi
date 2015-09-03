// bin/pdf-to-counts.cc

// Copyright 2012 Karel Vesely (Brno University of Technology)
//                Johns Hopkins University (author: Daniel Povey)

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

/** @brief Sums the pdf vectors to counts.  This is used to obtain priors
    for hybrid decoding.  */
    
#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Reads int32 vectors (same format as alignments, but typically\n"
        "actually representing pdfs, e.g. output by ali-to-pdf), and outputs\n"
        "counts for each index (typically one per per pdf), as a Vector<float>.\n"
        "\n"
        "Usage:  pdf-to-counts [options] <pdfs-rspecifier> <counts-wxfilname>\n"
        "e.g.: \n"
        " ali-to-pdf \"ark:gunzip -c 1.ali.gz|\" ark:- | \\\n"
        "   pdf-to-counts --binary=false ark:- counts.txt\n";
    ParseOptions po(usage);
    
    bool binary_write = false;
    po.Register("binary", &binary_write, "Write in binary mode");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string pdfs_rspecifier = po.GetArg(1),
        counts_wxfilename = po.GetArg(2);
    
    SequentialInt32VectorReader pdfs_reader(pdfs_rspecifier);

    std::vector<int64> counts; // will turn to Vector<BaseFloat> after counting.
    int32 num_done = 0;
    for (; !pdfs_reader.Done(); pdfs_reader.Next()) {
      std::vector<int32> alignment = pdfs_reader.Value();
      
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 value = alignment[i];
        if(value >= counts.size()) {
          counts.resize(value+1, 0);
        }
        counts[value]++; // accumulate counts
      }
      num_done++;
    }

    //convert to BaseFloat and write.
    Vector<BaseFloat> counts_f(counts.size());
    for(int32 i = 0; i < counts.size(); i++) {
      counts_f(i) = counts[i];
    }

    Output ko(counts_wxfilename, binary_write);
    counts_f.Write(ko.Stream(), binary_write);
    
    KALDI_LOG << "Summed " << num_done << " int32 vectors to counts, "
              << "total count is " << counts_f.Sum() << ", dim is "
              << counts_f.Dim();
    return (num_done == 0 ? 1 : 0); // error exit status if processed nothing.
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



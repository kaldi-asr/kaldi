// latbin/lattice-to-ngram-counts.cc

// Copyright 2014 Telepoint Global Hosting Service, LLC. (Author: David Snyder)
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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include <climits>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;

    const char *usage =
        "Retrieve n-label soft-counts for each input lattice. Each line of\n"
        "the output is of the form <uttid> <ngram_1>:<ngram_1-prob> ... "
        "<ngram_k>:<ngram_k-prob>.\n"
        "<ngram_k> is of the form <sym_1>,<sym_2>,...,<sym_n>.\n"
        "Note that <ngram_k> is an instance of that n-gram. The actual soft-\n"
        "counts are the consolidation of all instances of the same n-gram.\n"
        "Usage: lattice-to-ngram-counts [options] <lattice-rspecifier> "
        "<softcount-output-file>\n"
        " e.g.: lattice-to-ngram-counts --n=3 --eos-symbol=100 ark:lats "
        "counts.txt\n";
      
    ParseOptions po(usage);
    int32 n = 3;
    CompactLatticeArc::Label eos_symbol = INT_MAX;
    BaseFloat acoustic_scale = 0.075;

    std::string word_syms_filename;
    po.Register("n", &n, "n-gram context size for computing soft-counts");
    po.Register("eos-symbol", &eos_symbol, 
     "Integer label for the end of sentence character");
    po.Register("acoustic-scale", &acoustic_scale, 
     "Scaling factor for acoustic likelihoods");
    
    po.Read(argc, argv);
 
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    KALDI_ASSERT(n > 0);

    std::string lats_rspecifier = po.GetArg(1),
      softcount_wspecifier = po.GetOptArg(2);

    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    std::ofstream softcount_file;
    softcount_file.open(softcount_wspecifier.c_str()); 
    softcount_file.flush();

    int32 n_done = 0;
    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      KALDI_LOG << "Processing lattice for key " << key;
      CompactLattice lat = clat_reader.Value();
      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
      std::vector<std::pair<std::vector<CompactLattice::Arc::Label>, 
        double> > soft_counts;
      TopSortCompactLatticeIfNeeded(&lat);
      kaldi::ComputeSoftNgramCounts(lat, n, eos_symbol, &soft_counts);
      softcount_file << key << " ";
      for (int i = 0; i < soft_counts.size(); i++) {
        int32 size = soft_counts[i].first.size();
        for (int j = 0; j < size-1; j++) {
          softcount_file << soft_counts[i].first[j] << ",";
        }
        softcount_file << soft_counts[i].first[size-1] << ":"
                       << soft_counts[i].second << " ";
      }
      softcount_file << std::endl;
      clat_reader.FreeCurrent();
      n_done++;
    }
    KALDI_LOG << "Computed ngram soft counts for " << n_done 
              << " utterances.";
    softcount_file.close();
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

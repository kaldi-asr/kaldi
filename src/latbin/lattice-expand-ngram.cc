// latbin/lattice-expand-ngram.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using kaldi::CompactLatticeArc;

    const char *usage =
      "Expand lattices so that each arc has a unique n-label history, for\n"
      "a specified n (defaults to 3).\n"
      "Usage: lattice-expand-ngram [options] lattice-rspecifier "
      "lattice-wspecifier\n"
      "e.g.: lattice-expand-ngram --n=3 ark:lat ark:expanded_lat\n";

    ParseOptions po(usage);
    int32 n = 3;

    std::string word_syms_filename;
    po.Register("n", &n, "n-gram context to expand to.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(n > 0);

    std::string lats_rspecifier = po.GetArg(1),
      lats_wspecifier = po.GetOptArg(2);

    fst::UnweightedNgramFst<CompactLatticeArc> expand_fst(n);

    SequentialCompactLatticeReader lat_reader(lats_rspecifier);
    CompactLatticeWriter lat_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;

    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();
      KALDI_LOG << "Processing lattice for key " << key;
      CompactLattice lat = lat_reader.Value();
      CompactLattice expanded_lat;
      ComposeDeterministicOnDemand(lat, &expand_fst, &expanded_lat);
      if (expanded_lat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Empty lattice for utterance " << key;
       n_fail++;
      } else {
        if (lat.NumStates() == expanded_lat.NumStates()) {
          KALDI_LOG << "Lattice for key " << key
            << " did not need to be expanded for order " << n << ".";
        } else {
          KALDI_LOG << "Lattice expanded from " << lat.NumStates() << " to "
            << expanded_lat.NumStates() << " states for order " << n << ".";
        }
        lat_writer.Write(key, expanded_lat);
        n_done++;
      }
      lat_reader.FreeCurrent();
    }
    KALDI_LOG << "Processed " << n_done << " lattices with " << n_fail
      << " failures.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

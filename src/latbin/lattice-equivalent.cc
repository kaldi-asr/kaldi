// latbin/lattice-equivalent.cc

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Test whether sets of lattices are equivalent (return with status 0 if\n"
        "all were equivalent, 1 otherwise, -1 on error)\n"
        "Usage: lattice-equivalent [options] lattice-rspecifier1 lattice-rspecifier2\n"
        " e.g.: lattice-equivalent ark:1.lats ark:2.lats\n";
        
    ParseOptions po(usage);
    BaseFloat delta = 0.1; // Use a relatively high delta as for long paths, the absolute
    // scores can be quite large.
    int32 num_paths = 20;
    BaseFloat max_error_proportion = 0.0;
    po.Register("delta", &delta,
                "Delta parameter for equivalence test");
    po.Register("num-paths", &num_paths,
                "Number of paths per lattice for testing randomized equivalence");
    po.Register("max-error-proportion", &max_error_proportion,
                "Maximum proportion of missing 2nd lattices, or inequivalent "
                "lattices, we allow before returning nonzero status");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(max_error_proportion >= 0.0 && max_error_proportion <= 1.0);
    
    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2);

    // Read as regular lattice-- this is more efficient for testing
    // equivalence, I tihnk.
    SequentialLatticeReader lattice_reader1(lats_rspecifier1);

    RandomAccessLatticeReader lattice_reader2(lats_rspecifier2);
    

    int32 n_equivalent = 0, n_inequivalent = 0, n_no2nd = 0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      const Lattice &lat1 = lattice_reader1.Value();
      if (!lattice_reader2.HasKey(key)) {
        KALDI_WARN << "No 2nd lattice present for utterance " << key;
        n_no2nd++;
        continue;
      }
      const Lattice &lat2 = lattice_reader2.Value(key);
      if (fst::RandEquivalent(lat1, lat2, num_paths, delta, Rand())) {
        n_equivalent++;
        KALDI_LOG << "Lattices were equivalent for utterance " << key;
      } else {
        n_inequivalent++;
        KALDI_LOG << "Lattices were inequivalent for utterance " << key;
      }
    }
    KALDI_LOG << "Done " << (n_equivalent + n_inequivalent) << " lattices, "
              << n_equivalent << " were equivalent, " << n_inequivalent
              << " were not; for " << n_no2nd << ", could not find 2nd lattice."; 

    int32 num_inputs = n_equivalent + n_inequivalent + n_no2nd;
    int32 max_bad = max_error_proportion * num_inputs;
                
    if (n_no2nd > max_bad) return -1; // treat this as error.
    else return (n_inequivalent > max_bad ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

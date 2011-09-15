// latbin/lattice-compose.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
        "Takes two archives of lattices (indexed by utterances) and composes\n"
        "the individual lattice pairs (one from each archive).\n"
        "Does this using CompactLattice (acceptor) form.  If both lattices\n"
        "have alignments, will remove alignments from the first one.\n"
        "Usage: lattice-compose [options] lattice-rspecifier1 lattice-rspecifier2"
        " lattice-wspecifier\n"
        " e.g.: lattice-compose ark:1.lats ark:2.lats ark:composed.lats\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    SequentialCompactLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessCompactLatticeReader lattice_reader2(lats_rspecifier2);

    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_processed = 0, n_removed_ali = 0,
        n_empty = 0, n_success = 0, n_no_2ndlat=0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      CompactLattice lat1 = lattice_reader1.Value();
      lattice_reader1.FreeCurrent();
      if (lattice_reader2.HasKey(key)) {
        n_processed++;
        const CompactLattice& lat2 = lattice_reader2.Value(key);
        if (CompactLatticeHasAlignment(lat1) && CompactLatticeHasAlignment(lat2)) {
          RemoveAlignmentsFromCompactLattice(&lat1);
          n_removed_ali++;
        }
        CompactLattice lat3;
        Compose(lat1, lat2, &lat3);
        if (lat3.Start() == fst::kNoStateId) { // empty composition.
          KALDI_WARN << "For utterance " << key << ", composed result is empty.";
          n_empty++;
        } else {
          n_success++;
          compact_lattice_writer.Write(key, lat3);
        }
      } else {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats_rspecifier2 << ". Result of union will be the "
                   << "lattice found in " << lats_rspecifier1;
        n_no_2ndlat++;
      }
    }    
    KALDI_LOG << "Done " << n_processed << " lattices; "
              << n_success << " had nonempty result, " << n_empty
              << " had empty composition; in " << n_no_2ndlat
              << ", had empty second lattice.";
    return (n_success != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

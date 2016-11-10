// latbin/lattice-interp.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
        "Takes two archives of lattices (indexed by utterances) and combines\n"
        "the individual lattice pairs (one from each archive).  Keeps the alignments\n"
        "from the first lattice.  Equivalent to\n"
        "projecting the second archive on words (lattice-project), then composing\n"
        "the pairs of lattices (lattice-compose), then scaling graph and acoustic\n"
        "costs by 0.5 (lattice-scale).  You can control the individual scales with\n"
        "--alpha, which is the scale of the first lattices (the second is 1-alpha).\n"
        "Usage: lattice-interp [options] lattice-rspecifier-a lattice-rspecifier-b"
        " lattice-wspecifier\n"
        " e.g.: lattice-compose ark:1.lats ark:2.lats ark:composed.lats\n";

    ParseOptions po(usage);
    BaseFloat alpha = 0.5; // Scale of 1st in the pair.

    po.Register("alpha", &alpha, "Scale of the first lattice in the pair (should be in range [0, 1])");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    SequentialLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessCompactLatticeReader lattice_reader2(lats_rspecifier2);

    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_processed = 0, n_empty = 0, n_success = 0, n_no_2ndlat=0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      Lattice lat1 = lattice_reader1.Value();
      lattice_reader1.FreeCurrent();
      ScaleLattice(fst::LatticeScale(alpha, alpha), &lat1);
      ArcSort(&lat1, fst::OLabelCompare<LatticeArc>());

      if (lattice_reader2.HasKey(key)) {
        n_processed++;
        CompactLattice clat2 = lattice_reader2.Value(key);
        RemoveAlignmentsFromCompactLattice(&clat2);

        Lattice lat2;
        ConvertLattice(clat2, &lat2);
        fst::Project(&lat2, fst::PROJECT_OUTPUT); // project on words.
        ScaleLattice(fst::LatticeScale(1.0-alpha, 1.0-alpha), &lat2);
        ArcSort(&lat2, fst::ILabelCompare<LatticeArc>());

        Lattice lat3;
        Compose(lat1, lat2, &lat3);
        if (lat3.Start() == fst::kNoStateId) { // empty composition.
          KALDI_WARN << "For utterance " << key << ", composed result is empty.";
          n_empty++;
        } else {
          n_success++;
          CompactLattice clat3;
          ConvertLattice(lat3, &clat3);
          compact_lattice_writer.Write(key, clat3);
        }
      } else {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats_rspecifier2 << ". Not producing output";
        n_no_2ndlat++;
      }
    }
    KALDI_LOG << "Done " << n_processed << " lattices; "
              << n_success << " had nonempty result, " << n_empty
              << " had empty composition; in " << n_no_2ndlat
              << ", had empty second lattice.";
    return (n_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

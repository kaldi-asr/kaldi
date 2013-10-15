// latbin/lattice-union.cc

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
        "Takes two archives of lattices (indexed by utterances) and computes "
        "the union of the individual lattice pairs (one from each archive).\n"
        "Usage: lattice-union [options] lattice-rspecifier1 lattice-rspecifier2"
        " lattice-wspecifier\n"
        " e.g.: lattice-union ark:den.lats ark:num.lats ark:union.lats\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    SequentialLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessLatticeReader lattice_reader2(lats_rspecifier2);

    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_union = 0, n_no_lat = 0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      Lattice lat1 = lattice_reader1.Value();
      lattice_reader1.FreeCurrent();
      if (lattice_reader2.HasKey(key)) {
        const Lattice &lat2 = lattice_reader2.Value(key);
        Union(&lat1, lat2);
        n_union++;
      } else {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats_rspecifier2 << ". Result of union will be the "
                   << "lattice found in " << lats_rspecifier1;
        n_no_lat++;
      }

      Invert(&lat1);  // so that word labels are on the input.
      CompactLattice clat_out;
      DeterminizeLattice(lat1, &clat_out);
      // The determinization obviates the need to convert to conpact lattice
      // format using ConvertLattice(lat1, &clat_out);
      compact_lattice_writer.Write(key, clat_out);
      n_done++;
    }

    KALDI_LOG << "Total " << n_done << "lattices written. Computed union for "
              << n_union << " pairs of lattices. Missing second lattice in "
              << n_no_lat << " cases.";
    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

// latbin/lattice-topsort.cc

// Copyright 2017  Peter Smit

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

#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Topologically sort lattices.\n"
        "Fails if any of the lattices is malformed (e.g. has cycles)\n"
        "Usage: lattice-topsort [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-topsort ark:1.lats ark:1.sorted\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    int32 n_done = 0;
    SequentialCompactLatticeReader lattice_reader(lats_rspecifier);
    CompactLatticeWriter lattice_writer(lats_wspecifier);

    for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++) {
      CompactLattice clat = lattice_reader.Value();
      if (clat.Properties(fst::kTopSorted, false) == 0) {
        if (fst::TopSort(&clat) == false) {
          KALDI_ERR << "Cycles detected in lattice " << lattice_reader.Key() <<  " : cannot TopSort";
        }
      }
      lattice_writer.Write(lattice_reader.Key(), clat);
    }
    KALDI_LOG << "Done copying " << n_done << " lattices.";

    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
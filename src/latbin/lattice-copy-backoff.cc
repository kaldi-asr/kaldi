// latbin/lattice-copy-backoff.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Copy a table of lattices (1st argument), but for any keys that appear\n"
        "in the table from the 2nd argument, use the one from the 2nd argument.\n"
        "If the sets of keys are identical, this is equivalent to copying the 2nd\n"
        "table.  Note: the arguments are in this order due to the convention that\n"
        "sequential access is always over the 1st argument.\n"
        "\n"
        "Usage: lattice-copy-backoff [options] <lat-rspecifier1> "
        "<lat-rspecifier2> <lat-wspecifier>\n"
        " e.g.: lattice-copy-backoff ark:bad_but_complete.lat "
        "ark:good_but_incomplete.lat ark:out.lat\n";
      
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);
    
    int32 n_done = 0, n_backed_off = 0;
    
    SequentialCompactLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessCompactLatticeReader lattice_reader2(lats_rspecifier2);
    CompactLatticeWriter lattice_writer(lats_wspecifier);
    for (; !lattice_reader1.Done(); lattice_reader1.Next(), n_done++) {
      const std::string &key = lattice_reader1.Key();
      if (lattice_reader2.HasKey(key)) {
        lattice_writer.Write(key, lattice_reader2.Value(key));
      } else {
        lattice_writer.Write(key, lattice_reader1.Value());
        KALDI_VLOG(1) << "Backed off to 1st archive for key " << key;
        n_backed_off++;
      }
    }
    KALDI_LOG << "Done copying " << n_done << " lattices; backed off to 1st "
              << "archive for " << n_backed_off << " of those.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

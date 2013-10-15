// latbin/lattice-push.cc

// Copyright 2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/push-lattice.h"



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Push lattices, in CompactLattice format, so that the strings are as\n"
        "close to the start as possible, and the lowest cost weight for each\n"
        "state except the start state is (0, 0).  This can be helpful prior to\n"
        "word-alignment (in this case, only strings need to be pushed)\n"
        "\n"
        "Usage: lattice-push [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-push ark:1.lats ark:2.lats\n";

    ParseOptions po(usage);

    bool push_strings = true;
    bool push_weights = true;

    po.Register("push-strings", &push_strings, "If true, push the strings in the "
                "lattice to the start.");
    po.Register("push-weights", &push_weights, "If true, push the weights in the "
                "lattice to the start.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);


    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    CompactLatticeWriter clat_writer(lats_wspecifier); 

    int32 n_done = 0, n_err = 0;

    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      KALDI_VLOG(1) << "Processing lattice for utterance " << key;
      if (push_strings && !PushCompactLatticeStrings(&clat)) {
        KALDI_WARN << "Failure in pushing lattice strings (bad lattice?), "
                   << "for key " << key;
        n_err++;
        continue;
      }
      if (push_weights && !PushCompactLatticeWeights(&clat)) {
        KALDI_WARN << "Failure in pushing lattice weights (bad lattice?),"
                   << "for key " << key ;           
        n_err++;
        continue;
      }
      if (clat.NumStates() == 0) {
        KALDI_WARN << "Empty lattice for key " << key;
        n_err++;
        continue;
      }
      clat_writer.Write(key, clat);
      n_done++;
    }
    KALDI_LOG << "Pushed " << n_done << " lattices, errors on " << n_err;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

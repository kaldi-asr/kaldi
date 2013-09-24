// latbin/nbest-to-lattice.cc

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


bool GetUtteranceId(const std::string &nbest_id, std::string *utterance_id) {
  size_t pos = nbest_id.find_last_of('-');
  if (pos == std::string::npos || pos == 0) return false;
  else{
    *utterance_id = std::string(nbest_id, 0, pos);
    return true;
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Read in a Table containing N-best entries from a lattices (i.e. individual\n"
        "lattices with a linear structure, one for each N-best entry, indexed by\n"
        "utt_id_a-1, utt_id_a-2, etc., and take the union of them for each utterance\n"
        "id (e.g. utt_id_a), outputting a lattice for each.\n"
        "Usage:  nbest-to-lattice <nbest-rspecifier> <lattices-wspecifier>\n"
        " e.g.: nbest-to-lattice ark:1.nbest ark:1.lats\n";
      
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nbest_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);


    SequentialCompactLatticeReader compact_nbest_reader(nbest_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    int32 n_nbest_done = 0, n_utt_done = 0;

    // The variable "cur_union" will be the union of FSTs for a
    // particular utterance-id, if we have any "pending".
    CompactLattice cur_union;
    std::string cur_utt_id;

    for (; !compact_nbest_reader.Done(); compact_nbest_reader.Next()) {
      std::string nbest_id = compact_nbest_reader.Key();
      const CompactLattice &this_nbest = compact_nbest_reader.Value();
      std::string utt_id;
      if (!GetUtteranceId(nbest_id, &utt_id)) {
        KALDI_ERR << "Invalid n-best id " << nbest_id << ": make sure you "
            "are giving N-bests to nbest-to-lattice.";
      }
      if (utt_id != cur_utt_id) { // change in utterance.
        if (cur_utt_id != "") {
          compact_lattice_writer.Write(cur_utt_id, cur_union);
          cur_union.DeleteStates();
        }
        n_utt_done++; // We increment this when we start processing a
        // new utterance.
        cur_utt_id = utt_id;
      }
      Union(&cur_union, this_nbest);
      n_nbest_done++;
    }

    if (cur_utt_id != "")
      compact_lattice_writer.Write(cur_utt_id, cur_union);
    
    KALDI_LOG << "Done joining n-best into lattices for "
              << n_utt_done << " utterances, with on average "
              << (n_nbest_done/static_cast<BaseFloat>(n_utt_done))
              << " N-best paths per utterance.";
    return (n_utt_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

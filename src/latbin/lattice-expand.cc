// latbin/lattice-expand.cc

// Copyright 2021 Xiaomi Inc. (author: Daniel Povey)
//           2021 Johns Hopkins University (author: Ke Li)

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage = 
      "Expand lattices so that arcs with higher posteriors than epsilon have\n"
      "unique histories.\n"
      "Usage: lattice-expand [options] lattice-rspecifier lattice-wspecifier\n"
      "e.g.: lattice-expand --acoustic-scale=0.1 --epsilon=0.5 ark:lat ark:expanded_lat\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat inv_acoustic_scale = 1.0;
    BaseFloat epsilon = 0.1;

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for "
                "acoustic likelihoods.");
    po.Register("inv-acoustic-scale", &inv_acoustic_scale, "An alternative way "
                "of setting the acoustic scale: you can set its inverse.");
    po.Register("epsilon", &epsilon, "threshold of arc posteriors to control "
                "the size of the expanded lattice.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(acoustic_scale == 1.0 || inv_acoustic_scale == 1.0);
    if (inv_acoustic_scale != 1.0)
      acoustic_scale = 1.0 / inv_acoustic_scale;
    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";
    KALDI_ASSERT(epsilon > 0 && epsilon <= 1);

    std::string lats_rspecifier = po.GetArg(1),
      lats_wspecifier = po.GetArg(2);

    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    int32 n_done = 0, n_err = 0, n_expanded = 0;
    int64 n_arcs_in = 0, n_arcs_out = 0, n_states_in = 0, n_states_out = 0;
    
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
      int64 narcs = NumArcs(clat), nstates = clat.NumStates();
      n_arcs_in += narcs;
      n_states_in += nstates;
      CompactLattice expanded_clat;
      ExpandCompactLattice(clat, epsilon, &expanded_clat);
      if (expanded_clat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Empty lattice for utterance " << key;
        n_err++;
      } else {
        if (clat.NumStates() == expanded_clat.NumStates()) {
          n_arcs_out += narcs;
          n_states_out += nstates;
        } else {
          int64 expanded_narcs = NumArcs(expanded_clat),          
              expanded_nstates = expanded_clat.NumStates();
          n_arcs_out += expanded_narcs;
          n_states_out += expanded_nstates;
          n_expanded += 1;
        }
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale),
                          &expanded_clat);
        compact_lattice_writer.Write(key, expanded_clat);
        n_done++;
      }
    }
    
    BaseFloat den = (n_done > 0 ? static_cast<BaseFloat>(n_done) : 1.0);
    KALDI_LOG << "Overall, expanded from on average " << (n_states_in/den)
              << " to " << (n_states_out/den) << " states, and from "
              << (n_arcs_in/den) << " to " << (n_arcs_out/den) << " arcs, over "
              << n_done << " utterances.";
    KALDI_LOG << "Overall, " << static_cast<BaseFloat>(n_expanded)/n_done * 100
              << "\% percentage of lattices get expanded.";
    KALDI_LOG << "Processed " << n_done << " lattices with " << n_err
              << " failures.";
    return (n_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

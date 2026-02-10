// latbin/lattice-prune.cc

// Copyright 2009-2013  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Apply beam pruning to lattices\n"
        "Usage: lattice-prune [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-prune --acoustic-scale=0.1 --beam=4.0 ark:1.lats ark:pruned.lats\n";
      
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat inv_acoustic_scale = 1.0;
    BaseFloat beam = 10.0;
    
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("inv-acoustic-scale", &inv_acoustic_scale, "An alternative way of setting the "
                "acoustic scale: you can set its inverse.");
    po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling]");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(acoustic_scale == 1.0 || inv_acoustic_scale == 1.0);
    if (inv_acoustic_scale != 1.0)
      acoustic_scale = 1.0 / inv_acoustic_scale;
    
    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);


    
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    int32 n_done = 0, n_err = 0;
    int64 n_arcs_in = 0, n_arcs_out = 0,
        n_states_in = 0, n_states_out = 0;

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";
    
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
      int64 narcs = NumArcs(clat), nstates = clat.NumStates();
      n_arcs_in += narcs;
      n_states_in += nstates;
      CompactLattice pruned_clat(clat);
      if (!PruneLattice(beam, &pruned_clat)) {
        KALDI_WARN << "Error pruning lattice for utterance " << key;
        n_err++;
      }
      int64 pruned_narcs = NumArcs(pruned_clat),          
          pruned_nstates = pruned_clat.NumStates();
      n_arcs_out += pruned_narcs;
      n_states_out += pruned_nstates;
      KALDI_LOG << "For utterance " << key << ", pruned #states from "
                << nstates << " to " << pruned_nstates << " and #arcs from "
                << narcs << " to " << pruned_narcs;
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &pruned_clat);
      compact_lattice_writer.Write(key, pruned_clat);
      n_done++;
    }

    BaseFloat den = (n_done > 0 ? static_cast<BaseFloat>(n_done) : 1.0);
    KALDI_LOG << "Overall, pruned from on average " << (n_states_in/den) << " to "
              << (n_states_out/den) << " states, and from " << (n_arcs_in/den)
              << " to " << (n_arcs_out/den) << " arcs, over " << n_done
              << " utterances.";
    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

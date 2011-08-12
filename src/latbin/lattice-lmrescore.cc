// latbin/lattice-lmrescore.cc

// Copyright 2009-2011  Microsoft Corporation

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
        "Add lm_scale * [cost of best path through LM FST] to graph-cost of\n"
        "paths through lattice.  Does this by composing with LM FST, then\n"
        "lattice-determinizing (it has to negate weights first if lm_scale<0)\n"
        "Usage: lattice-lmrescore [options] lattice-rspecifier lm-fst-in lattice-wspecifier\n"
        " e.g.: lattice-lmrescore --lm-scale=-1.0 ark:in.lats ark:out.lats\n";
      
    ParseOptions po(usage);
    BaseFloat lm_scale = 1.0;
    
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model costs");
    po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling]");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    VectorFst<StdArc> *lm_fst = NULL; 
    {
      Input ki(fst_rxfilename);
      lm_fst = VectorFst<StdArc>::Read(
          ki.Stream(),
          fst::FstReadOptions((std::string)fst_rxfilename));
      if (lm_fst == NULL)
        exit(1);
    }

    // mapped_fst is the LM fst interpreted using the LatticeWeight semiring,
    // with all the cost on the first member of the pair (since it's a graph
    // weight).
    fst::LatticeToStdMapper mapper;
    MapFst<StdArc, LatticeArc, LatticeToStdMapper> mapped_fst(lm_fst, mapper);
    
    // Read as regular lattice-- this is the form we need it in for efficient
    // composition and determinization.
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    int32 n_done = 0; // there is no failure mode, barring a crash.
    int64 n_arcs_in = 0, n_arcs_out = 0,
        n_states_in = 0, n_states_out = 0;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      if (lm_scale != 0.0) {
        // Only need to modify it if LM scale nonzero.
        if (


      }

      
      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
      int64 narcs = NumArcs(lat), nstates = lat.NumStates();
      n_arcs_in += narcs;
      n_states_in += nstates;
      Lattice pruned_lat;
      Prune(lat, &pruned_lat, beam_weight);
      int64 pruned_narcs = NumArcs(pruned_lat),
          pruned_nstates = pruned_lat.NumStates();
      n_arcs_out += pruned_narcs;
      n_states_out += pruned_nstates;
      KALDI_LOG << "For utterance " << key << ", pruned #states from "
                << nstates << " to " << pruned_nstates << " and #arcs from"
                << narcs << " to " << pruned_narcs;
      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &pruned_lat);
      CompactLattice pruned_clat;
      ConvertLattice(pruned_lat, &pruned_clat);
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
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

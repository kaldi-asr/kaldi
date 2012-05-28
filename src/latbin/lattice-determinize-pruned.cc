// latbin/lattice-determinize-pruned.cc

// Copyright 2012  Daniel Povey

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
#include "lat/kaldi-lattice.h"
#include "fstext/determinize-lattice-pruned.h"

namespace kaldi {

bool DeterminizeLatticeWrapper(const Lattice &lat,
                               const std::string &key,
                               bool prune,
                               BaseFloat beam,
                               BaseFloat beam_ratio,
                               int32 max_mem,
                               int32 max_loop,
                               BaseFloat delta,
                               int32 num_loops,
                               CompactLattice *clat) {
  fst::DeterminizeLatticeOptions lat_opts;
  lat_opts.max_mem = max_mem;
  lat_opts.max_loop = max_loop;
  lat_opts.delta = delta;
  BaseFloat cur_beam = beam;
  for (int32 i = 0; i < num_loops;) { // we increment i below.

    if (lat.Start() == fst::kNoStateId) {
      KALDI_WARN << "Detected empty lattice, skipping " << key;
      return false;
    }
    
    // The work gets done in the next line.  
    if (DeterminizeLattice(lat, clat, lat_opts, NULL)) { 
      if (prune)
        fst::PruneCompactLattice(LatticeWeight(cur_beam, 0), clat);
      return true;
    } else { // failed to determinize..
      KALDI_WARN << "Failed to determinize lattice (presumably max-states "
                 << "reached), reducing lattice-beam to "
                 << (cur_beam*beam_ratio) << " and re-trying.";
      for (; i < num_loops; i++) {
        cur_beam *= beam_ratio;
        Lattice pruned_lat(lat);
        Prune(lat, &pruned_lat, LatticeWeight(cur_beam, 0));
        if (NumArcs(lat) == NumArcs(pruned_lat)) {
          cur_beam *= beam_ratio;
          KALDI_WARN << "Pruning did not have an effect on the original "
                     << "lattice size; reducing beam to "
                     << cur_beam << " and re-trying.";
        } else if (DeterminizeLattice(pruned_lat, clat, lat_opts, NULL)) {
          if (prune)
            fst::PruneCompactLattice(LatticeWeight(cur_beam, 0), clat);
          return true;
        } else {
          KALDI_WARN << "Determinization failed again; reducing beam again to "
                     << (cur_beam*beam_ratio) << " and re-trying.";
        }
      }
    }
  }
  KALDI_WARN << "Decreased pruning beam --num-loops=" << num_loops
             << " times and was not able to determinize: failed for "
             << key;
  return false;
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    
    const char *usage =
        "Determinize lattices, keeping only the best path (sequence of acoustic states)\n"
        "for each input-symbol sequence.  This version does pruning as part of the\n"
        "determinization algorithm, which is more efficient and prevents blowup.\n"
        "See http://kaldi.sourceforge.net/lattices.html for more information on lattices.\n"
        "\n"
        "Usage: lattice-determinize-pruned [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-determinize-pruned --acoustic-scale=0.1 --beam=6.0 ark:in.lats ark:det.lats\n";
    
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat beam = 10.0;
    fst::DeterminizeLatticePrunedOptions opts; // Options used in DeterminizeLatticePruned--
    // this options class does not have its own Register function as it's viewed as
    // being more part of "fst world", so we register its elements independently.
    opts.max_mem = 50000000;
    opts.max_loop = 500000;
    
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling].");
    po.Register("delta", &opts.delta, "Tolerance used in determinization");
    po.Register("max-mem", &opts.max_mem, "Maximum approximate memory usage in "
                "determinization (real usage might be many times this)");
    po.Register("max-arcs", &opts.max_arcs, "Maximum number of arcs in output FST (total, "
                "not per state");
    po.Register("max-states", &opts.max_states, "Maximum number of arcs in output FST (total, "
                "not per state");
    po.Register("max-loop", &opts.max_loop, "Option used to detect a particular type of determinization "
                "failure, typically due to invalid input (e.g., negative-cost loops)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);


    // Read as regular lattice-- this is the form the determinization code
    // accepts.
    SequentialLatticeReader lat_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_lat_writer(lats_wspecifier); 

    int32 n_done = 0, n_warn = 0;

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";
    LatticeWeight beam_weight(beam, static_cast<BaseFloat>(0.0));
    
    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();
      Lattice lat = lat_reader.Value();
      Invert(&lat); // so word labels are on the input side.
      lat_reader.FreeCurrent();
      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
      if (!TopSort(&lat)) {
        KALDI_WARN << "Could not topologically sort lattice: this probably means it"
            " has bad properties e.g. epsilon cycles.  Your LM or lexicon might "
            "be broken, e.g. LM with epsilon cycles or lexicon with empty words.";
      }
      fst::ArcSort(&lat, fst::ILabelCompare<LatticeArc>());
      CompactLattice det_clat;
      if (!DeterminizeLatticePruned(lat, beam_weight, &det_clat, opts)) {
        KALDI_WARN << "For key " << key << ", determinization did not succeed"
            "(partial output will be pruned tighter than the specified beam.)";
        n_warn++;
      }
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &det_clat);
      compact_lat_writer.Write(key, det_clat);
      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " lattices, determinization finished "
              << "earlier than specified by the beam on " << n_warn
              << " of these.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

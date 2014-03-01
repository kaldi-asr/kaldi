// latbin/lattice-determinize-phone-pruned.cc

// Copyright 2014  Guoguo Chen

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
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "lat/push-lattice.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    
    const char *usage =
        "Determinize lattices, keeping only the best path (sequence of\n"
        "acoustic states) for each input-symbol sequence. This version does\n"
        "phone inertion when doing a first pass determinization, it then\n"
        "removes the inserted symbols and does a second pass determinization.\n"
        "It also does pruning as part of the determinization algorithm, which\n"
        "is more efficient and prevents blowup.\n"
        "\n"
        "Usage: lattice-determinize-phone-pruned [options] <model> \\\n"
        "                  <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-determinize-phone-pruned --acoustic-scale=0.1 \\\n"
        "                            final.mdl ark:in.lats ark:det.lats\n";
    
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat beam = 10.0;
    fst::DeterminizeLatticePhonePrunedOptions opts;
    opts.max_mem = 50000000;
    
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic"
                " likelihoods.");
    po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling].");
    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // Reads as regular lattice-- this is the form the determinization code
    // accepts.
    SequentialLatticeReader lat_reader(lats_rspecifier);
    
    // Writes as compact lattice.
    CompactLatticeWriter compact_lat_writer(lats_wspecifier); 

    int32 n_done = 0, n_warn = 0;

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();
      Lattice lat = lat_reader.Value();
      lat_reader.FreeCurrent();

      KALDI_VLOG(2) << "Processing lattice " << key;

      fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);

      CompactLattice det_clat;
      if (!DeterminizeLatticePhonePrunedWrapper(
              trans_model, &lat, beam, &det_clat, opts)) {
        KALDI_WARN << "For key " << key << ", determinization did not succeed"
            "(partial output will be pruned tighter than the specified beam.)";
        n_warn++;
      }

      fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &det_clat);
      compact_lat_writer.Write(key, det_clat);
      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " lattices, determinization finished "
              << "earlier than specified by the beam on " << n_warn << " of "
              << "these.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

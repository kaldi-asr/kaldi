// latbin/lattice-to-post.cc

// Copyright 2009-2011   Saarland University
// Author: Arnab Ghoshal

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
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Do forward-backward and collect posteriors over lattices.\n"
        "Usage: lattice-to-post [options] lats-rspecifier posts-wspecifier [loglikes-wspecifier]\n"
        " e.g.: lattice-to-post --acoustic-scale=0.1 ark:1.lats ark:1.post\n";

    kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    kaldi::ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    std::string lats_rspecifier = po.GetArg(1),
        posteriors_wspecifier = po.GetArg(2),
        loglikes_wspecifier = po.GetOptArg(3);

    // Read as regular lattice
    kaldi::SequentialLatticeReader lattice_reader(lats_rspecifier);

    kaldi::PosteriorWriter posterior_writer(posteriors_wspecifier);
    kaldi::BaseFloatWriter loglikes_writer(loglikes_wspecifier);

    int32 n_done = 0;
    double total_like = 0.0, lat_like;
    double total_ac_like = 0.0, lat_ac_like; // acoustic likelihood weighted by posterior.
    double total_time = 0, lat_time;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      kaldi::Lattice lat = lattice_reader.Value();
      // FreeCurrent() is an optimization that prevents the lattice from being
      // copied unnecessarily (OpenFst does copy-on-write).
      lattice_reader.FreeCurrent();
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &lat);
      
      kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }

      kaldi::Posterior post;
      lat_like = kaldi::LatticeForwardBackward(lat, &post, &lat_ac_like);
      total_like += lat_like;
      lat_time = post.size();
      total_time += lat_time;
      total_ac_like += lat_ac_like;

      KALDI_VLOG(2) << "Processed lattice for utterance: " << key << "; found "
                    << lat.NumStates() << " states and " << fst::NumArcs(lat)
                    << " arcs. Average log-likelihood = " << (lat_like/lat_time)
                    << " over " << lat_time << " frames.  Average acoustic log-like"
                    << " per frame is " << (lat_ac_like/lat_time);
      
      if (loglikes_writer.IsOpen()) 
        loglikes_writer.Write(key, lat_like);

      posterior_writer.Write(key, post);
      n_done++;
    }

    KALDI_LOG << "Overall average log-like/frame is "
              << (total_like/total_time) << " over " << total_time
              << " frames.  Average acoustic like/frame is "
              << (total_ac_like/total_time);
    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

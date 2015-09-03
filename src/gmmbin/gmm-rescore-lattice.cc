// gmmbin/gmm-rescore-lattice.cc

// Copyright 2009-2011   Saarland University (Author: Arnab Ghoshal)
//                       Cisco Systems (Author: Neha Agrawal)

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
#include "util/stl-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "gmm/decodable-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Replace the acoustic scores on a lattice using a new model.\n"
        "Usage: gmm-rescore-lattice [options] <model-in> <lattice-rspecifier> "
        "<feature-rspecifier> <lattice-wspecifier>\n"
        " e.g.: gmm-rescore-lattice 1.mdl ark:1.lats scp:trn.scp ark:2.lats\n";

    kaldi::BaseFloat old_acoustic_scale = 0.0;
    kaldi::ParseOptions po(usage);
    po.Register("old-acoustic-scale", &old_acoustic_scale,
                "Add in the scores in the input lattices with this scale, rather "
                "than discarding them.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    // Read as regular lattice
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_frames = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      if (!feature_reader.HasKey(key)) {
        KALDI_WARN << "No feature found for utterance " << key << ". Skipping";
        num_err++;
        continue;
      }

      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      if (old_acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &clat);

      const Matrix<BaseFloat> &feats = feature_reader.Value(key);

      DecodableAmDiagGmm gmm_decodable(am_gmm, trans_model, feats);
      if (kaldi::RescoreCompactLattice(&gmm_decodable, &clat)) {
        compact_lattice_writer.Write(key, clat);
        num_done++;
        num_frames += feats.NumRows();
      } else num_err++;
    }

    KALDI_LOG << "Done " << num_done << " lattices with errors on "
              << num_err << ", #frames is " << num_frames;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

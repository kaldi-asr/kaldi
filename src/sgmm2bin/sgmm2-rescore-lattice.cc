// sgmm2bin/sgmm2-rescore-lattice.cc

// Copyright 2009-2012   Saarland University (Author: Arnab Ghoshal)
//                       Johns Hopkins University (Author: Daniel Povey)
//                       Cisco Systems (Author: Neha Agrawal)

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
#include "sgmm2/am-sgmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace kaldi {

struct Tuple {
  Tuple(int32 state, int32 arc, int32 offset):
    state_id(state), arc_id(arc), trans_offset(offset) {}
  int32 state_id;
  int32 arc_id;
  int32 trans_offset;
};

void LatticeAcousticRescore(const AmSgmm2 &am,
                            const TransitionModel &trans_model,
                            const MatrixBase<BaseFloat> &data,
                            const std::vector<std::vector<int32> > &gselect,
                            double log_prune,
                            const std::vector<int32> state_times,
                            Sgmm2PerSpkDerivedVars *spk_vars,
                            CompactLattice *clat) {
  kaldi::uint64 props = clat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());

  std::vector<std::vector< Tuple > > time_to_state(data.NumRows());

  for (size_t state = 0; state < state_times.size(); state++) {
    KALDI_ASSERT(state_times[state] >= 0);

    int32 t = state_times[state];
    int32 arc_id = 0;
    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
        !aiter.Done(); aiter.Next()) {
      CompactLatticeArc arc = aiter.Value();
      std::vector<int32> arc_string = (arc.weight.String());

      for (size_t offset = 0; offset < arc_string.size(); offset++) {
        if (t < data.NumRows()) // end state may be past this..
          time_to_state[t+offset].push_back(Tuple(state, arc_id, offset));
        else
          KALDI_ASSERT(t == data.NumRows()
                && "There appears to be lattice/feature mismatch.");
      }
      arc_id++;
    }
    if (clat->Final(state) != CompactLatticeWeight::Zero()) {
      std::vector<int32> arc_string = clat->Final(state).String();
      for (size_t offset = 0; offset < arc_string.size(); offset++) {
        if (t < data.NumRows()) // end state may be past this..
          time_to_state[t+offset].push_back(Tuple(state, -1, offset));
        else
          KALDI_ASSERT(t == data.NumRows()
                && "There appears to be lattice/feature mismatch.");
      }
    }
  }

  Sgmm2LikelihoodCache like_cache(am.NumGroups(), am.NumPdfs());

  for (int32 t = 0; t < data.NumRows(); t++, like_cache.NextFrame()) {
      Sgmm2PerFrameDerivedVars per_frame_vars;
      am.ComputePerFrameVars(data.Row(t), gselect[t], *spk_vars,
                             &per_frame_vars);

      for (size_t i = 0; i < time_to_state[t].size(); i++) {
        int32 state = time_to_state[t][i].state_id;
        int32 arc_id = time_to_state[t][i].arc_id;
        int32 offset = time_to_state[t][i].trans_offset;

        if (arc_id == -1) { // Final state
          // Access the trans_id
          CompactLatticeWeight curr_clat_weight = clat->Final(state);
          int32 trans_id = curr_clat_weight.String()[offset];

          // Calculate likelihood
          if (trans_id != 0) {  // Non-epsilon input label on arc
            int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
            BaseFloat ll = am.LogLikelihood(per_frame_vars, pdf_id,
                                            &like_cache, spk_vars, log_prune);

            // update weight
            CompactLatticeWeight new_clat_weight = curr_clat_weight;
            LatticeWeight new_lat_weight = new_clat_weight.Weight();
            new_lat_weight.SetValue2(-ll + curr_clat_weight.Weight().Value2());
            new_clat_weight.SetWeight(new_lat_weight);

            clat->SetFinal(state, new_clat_weight);
          }

        } else {
          fst::MutableArcIterator<CompactLattice> aiter(clat, state);

          // Access the trans_id
          aiter.Seek(arc_id);
          CompactLatticeArc arc = aiter.Value();
          int32 trans_id = arc.weight.String()[offset];

          // Calculate likelihood
          if (trans_id != 0) {  // Non-epsilon input label on arc
            int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
            BaseFloat ll = am.LogLikelihood(per_frame_vars, pdf_id,
                                            &like_cache, spk_vars, log_prune);
            // update weight
            LatticeWeight new_weight = arc.weight.Weight();
            new_weight.SetValue2(-ll + arc.weight.Weight().Value2());
            arc.weight.SetWeight(new_weight);
            aiter.SetValue(arc);
          }
        }
      }
  }
}

}  // namespace kaldi

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
      "Usage: sgmm2-rescore-lattice [options] <model-in> <lattice-rspecifier> "
      "<feature-rspecifier> <lattice-wspecifier>\n"
      " e.g.: sgmm2-rescore-lattice 1.mdl ark:1.lats scp:trn.scp ark:2.lats\n";

    kaldi::BaseFloat old_acoustic_scale = 0.0;
    BaseFloat log_prune = 5.0;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;

    kaldi::ParseOptions po(usage);
    po.Register("old-acoustic-scale", &old_acoustic_scale,
                "Add the current acoustic scores with some scale.");
    po.Register("log-prune", &log_prune,
                "Pruning beam used to reduce number of exp() evaluations.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("gselect", &gselect_rspecifier,
                "Precomputed Gaussian indices (rspecifier)");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect-rspecifier option is required.";

    std::string model_filename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    AmSgmm2 am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    // Read as compact lattice
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string utt = compact_lattice_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        KALDI_WARN << "No feature found for utterance " << utt;
        num_err++;
        continue;
      }

      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      if (old_acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &clat);

      kaldi::uint64 props = clat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&clat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }

      vector<int32> state_times;
      int32 max_time = kaldi::CompactLatticeStateTimes(clat, &state_times);
      const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
      if (feats.NumRows() != max_time) {
        KALDI_WARN << "Skipping utterance " << utt << " since number of time "
                   << "frames in lattice ("<< max_time << ") differ from "
                   << "number of feature frames (" << feats.NumRows() << ").";
        num_err++;
        continue;
      }

      // Get speaker vectors
      Sgmm2PerSpkDerivedVars spk_vars;
      if (spkvecs_reader.IsOpen()) {
        if (spkvecs_reader.HasKey(utt)) {
          spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
          am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
        } else {
          KALDI_WARN << "Cannot find speaker vector for " << utt;
          num_err++;
          continue;
        }
      }  // else spk_vars is "empty"

      if (!gselect_reader.HasKey(utt) ||
          gselect_reader.Value(utt).size() != feats.NumRows()) {
        KALDI_WARN << "No Gaussian-selection info available for utterance "
                   << utt << " (or wrong size)";
        num_err++;
        continue;
      }
      const std::vector<std::vector<int32> > &gselect =
          gselect_reader.Value(utt);

      kaldi::LatticeAcousticRescore(am_sgmm, trans_model, feats,
                                    gselect, log_prune,
                                    state_times, &spk_vars, &clat);

      compact_lattice_writer.Write(utt, clat);
      num_done++;
    }

    KALDI_LOG << "Done " << num_done << " lattices, errors on "
              << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

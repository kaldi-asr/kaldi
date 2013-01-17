// sgmmbin/sgmm-rescore-lattice.cc

// Copyright 2009-2011   Saarland University
// Author: Arnab Ghoshal

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
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace kaldi {

void LatticeAcousticRescore(const AmSgmm &am,
                            const TransitionModel &trans_model,
                            const MatrixBase<BaseFloat> &data,
                            const SgmmPerSpkDerivedVars &spk_vars,
                            const std::vector<std::vector<int32> > &gselect,
                            const SgmmGselectConfig &sgmm_config,
                            double log_prune,
                            const std::vector<int32> state_times,
                            Lattice *lat) {
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());
  std::vector<std::vector<int32> > time_to_state(data.NumRows());
  for (size_t i = 0; i < state_times.size(); i++) {
    KALDI_ASSERT(state_times[i] >= 0);
    if (state_times[i] < data.NumRows()) // end state may be past this..
      time_to_state[state_times[i]].push_back(i);
    else
      KALDI_ASSERT(state_times[i] == data.NumRows()
                   && "There appears to be lattice/feature mismatch.");
  }
  
  for (int32 t = 0; t < data.NumRows(); t++) {
    SgmmPerFrameDerivedVars per_frame_vars;
    std::vector<int32> this_gselect;
    if (!gselect.empty()) {
      KALDI_ASSERT(t < gselect.size());
      this_gselect = gselect[t];
    } else  {
      am.GaussianSelection(sgmm_config, data.Row(t), &this_gselect);
    }
    am.ComputePerFrameVars(data.Row(t), this_gselect, spk_vars,
                           0.0 /*fMLLR logdet*/, &per_frame_vars);
                           
    unordered_map<int32, BaseFloat> pdf_id_to_like;
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
           aiter.Next()) {
        LatticeArc arc = aiter.Value();
        int32 trans_id = arc.ilabel;
        if (trans_id != 0) {  // Non-epsilon input label on arc
          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
          BaseFloat ll;
          if (pdf_id_to_like.count(pdf_id) == 0) {
            ll = am.LogLikelihood(per_frame_vars, pdf_id, log_prune);
            pdf_id_to_like[pdf_id] = ll;
          } else {
            ll = pdf_id_to_like[pdf_id];
          }
          arc.weight.SetValue2(-ll + arc.weight.Value2());
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
        "Usage: sgmm-rescore-lattice [options] <model-in> <lattice-rspecifier> "
        "<feature-rspecifier> <lattice-wspecifier>\n"
        " e.g.: sgmm-rescore-lattice 1.mdl ark:1.lats scp:trn.scp ark:2.lats\n";

    kaldi::BaseFloat old_acoustic_scale = 0.0;
    BaseFloat log_prune = 5.0;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    SgmmGselectConfig sgmm_opts;    
    kaldi::ParseOptions po(usage);
    po.Register("old-acoustic-scale", &old_acoustic_scale,
                "Add the current acoustic scores with some scale.");
    po.Register("log-prune", &log_prune, "Pruning beam used to reduce number of exp() evaluations.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    sgmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    AmSgmm am_sgmm;
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
    // Read as regular lattice
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); 

    int32 n_done = 0, num_no_feats = 0, num_other_error = 0;
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string utt = lattice_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        KALDI_WARN << "No feature found for utterance " << utt << ". Skipping";
        num_no_feats++;
        continue;
      }

      Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      if (old_acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &lat);

      kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      } 

      vector<int32> state_times;
      int32 max_time = kaldi::LatticeStateTimes(lat, &state_times);
      const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
      if (feats.NumRows() != max_time) {
        KALDI_WARN << "Skipping utterance " << utt << " since number of time "
                   << "frames in lattice ("<< max_time << ") differ from "
                   << "number of feature frames (" << feats.NumRows() << ").";
        num_other_error++;
        continue;
      }

      // Get speaker vectors      
      SgmmPerSpkDerivedVars spk_vars;
      if (spkvecs_reader.IsOpen()) {
        if (spkvecs_reader.HasKey(utt)) {
          spk_vars.v_s = spkvecs_reader.Value(utt);
          am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
        } else {
          KALDI_WARN << "Cannot find speaker vector for " << utt;
          num_other_error++;
          continue;
        }
      }  // else spk_vars is "empty"

      bool have_gselect  = !gselect_rspecifier.empty()
          && gselect_reader.HasKey(utt)
          && gselect_reader.Value(utt).size() == feats.NumRows();
      if (!gselect_rspecifier.empty() && !have_gselect)
        KALDI_WARN << "No Gaussian-selection info available for utterance "
                   << utt << " (or wrong size)";
      std::vector<std::vector<int32> > empty_gselect;
      const std::vector<std::vector<int32> > *gselect =
          (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);
        
      kaldi::LatticeAcousticRescore(am_sgmm, trans_model, feats,
                                    spk_vars, *gselect, sgmm_opts,
                                    log_prune, state_times, &lat);
      CompactLattice clat_out;
      ConvertLattice(lat, &clat_out);
      compact_lattice_writer.Write(utt, clat_out);
      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

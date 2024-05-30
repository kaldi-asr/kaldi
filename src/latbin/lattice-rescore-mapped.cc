// latbin/lattice-rescore-mapped.cc

// Copyright 2009-2012   Saarland University (author: Arnab Ghoshal)
//                       Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace kaldi {

void LatticeAcousticRescore(const TransitionModel &trans_model,
                            const Matrix<BaseFloat> &log_likes,
                            const std::vector<int32> &state_times,
                            Lattice *lat) {
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());
  std::vector<std::vector<int32> > time_to_state(log_likes.NumRows());
  for (size_t i = 0; i < state_times.size(); i++) {
    KALDI_ASSERT(state_times[i] >= 0);
    if (state_times[i] < log_likes.NumRows()) // end state may be past this..
      time_to_state[state_times[i]].push_back(i);
    else
      KALDI_ASSERT(state_times[i] == log_likes.NumRows()
                   && "There appears to be lattice/feature mismatch.");
  }

  for (int32 t = 0; t < log_likes.NumRows(); t++) {
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
           aiter.Next()) {
        LatticeArc arc = aiter.Value();
        int32 trans_id = arc.ilabel;
        if (trans_id != 0) {  // Non-epsilon input label on arc
          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
          if (pdf_id > log_likes.NumCols())
            KALDI_ERR << "Pdf-id " << pdf_id << " is out of the range of "
                      << "input log-likelihoods " << log_likes.NumCols()
                      << " (probably some kind of mismatch).";
          BaseFloat ll = log_likes(t, pdf_id);
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
        "Replace the acoustic scores on a lattice using log-likelihoods read in\n"
        "as a matrix for each utterance, indexed (frame, pdf-id).  This does the same\n"
        "as (e.g.) gmm-rescore-lattice, but from a matrix.  The \"mapped\" means that\n"
        "the transition-model is used to map transition-ids to pdf-ids.  (c.f.\n"
        "latgen-faster-mapped).  Note: <transition-model-in> can be any type of\n"
        "model file, e.g. GMM-based or neural-net based; only the transition model is read.\n"
        "\n"
        "Usage: lattice-rescore-mapped [options] <transition-model-in> <lattice-rspecifier> "
        "<loglikes-rspecifier> <lattice-wspecifier>\n"
        " e.g.: nnet-logprob [args] .. | lattice-rescore-mapped final.mdl ark:1.lats ark:- ark:2.lats\n";

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
        loglike_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      // Ignore what follows it in the model.
    }

    RandomAccessBaseFloatMatrixReader loglike_reader(loglike_rspecifier);
    // Read as regular lattice
    SequentialLatticeReader lattice_reader(lats_rspecifier);
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_frames = 0;
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      if (!loglike_reader.HasKey(key)) {
        KALDI_WARN << "No log-likes found for utterance " << key << ". Skipping";
        num_err++;
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

      std::vector<int32> state_times;
      int32 max_time = kaldi::LatticeStateTimes(lat, &state_times);
      const Matrix<BaseFloat> &log_likes = loglike_reader.Value(key);
      if (log_likes.NumRows() != max_time) {
        KALDI_WARN << "Skipping utterance " << key << " since number of time "
                   << "frames in lattice ("<< max_time << ") differ from "
                   << "number of frames in log-likelihoods (" << log_likes.NumRows() << ").";
        num_err++;
        continue;
      }

      kaldi::LatticeAcousticRescore(trans_model, log_likes, state_times,
                                    &lat);
      CompactLattice clat_out;
      ConvertLattice(lat, &clat_out);
      compact_lattice_writer.Write(key, clat_out);
      num_done++;
      num_frames += log_likes.NumRows();
    }

    KALDI_LOG << "Done " << num_done << " lattices, " << num_err
              << " with errors, #frames is " << num_frames;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

// durmodbin/nnet3-durmodel-rescore-lattice.cc

// Copyright 2015 Hossein Hadian

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
#include "hmm/transition-model.h"
#include "durmod/kaldi-durmod.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using kaldi::CompactLatticeArc;

    const char *usage =
      "Rescore a lattice using the scores from a phone duration model.\n"
      "Usage: nnet3-durmodel-rescore-lattice [options] <nnet3-dur-model> "
      "<trans-model> <lattice-rspecifier> <lattice-wspecifier>\n"
      "e.g.: \n"
      "nnet3-durmodel-rescore-lattice 20.mdl final.mdl "
      "ark:lat.1 ark:rescored_lat.1\n";

    BaseFloat duration_model_scale = 1.0;
    std::string avg_logprobs_file;
    ParseOptions po(usage);
    po.Register("duration-model-scale", &duration_model_scale, "Scaling factor "
                "for duration model costs");
    po.Register("avg-logprobs-file", &avg_logprobs_file, "File containing avg logprobs to be subtracted from scores.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    TransitionModel trans_model;
    std::string nnet_durmodel_filename = po.GetArg(1),
                model_filename = po.GetArg(2),
                lats_rspecifier = po.GetArg(3),
                lats_wspecifier = po.GetOptArg(4);

    ReadKaldiObject(model_filename, &trans_model);
    NnetPhoneDurationModel nnet_durmodel;
    ReadKaldiObject(nnet_durmodel_filename, &nnet_durmodel);
    
    /*//tmp// AvgPhoneLogProbs avg_logprobs;
    if (!avg_logprobs_file.empty()) {
      ReadKaldiObject(avg_logprobs_file, &avg_logprobs);
    }*/
    Vector<BaseFloat> priors;
    if (!avg_logprobs_file.empty()) {
      bool binary_in;
      Input ki(avg_logprobs_file, &binary_in);
      priors.Read(ki.Stream(), binary_in, false);
    }
    NnetPhoneDurationScoreComputer durmodel_scorer(nnet_durmodel, priors);

    // Read and write as compact lattice.
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, n_fail = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      KALDI_LOG << "Rescoring lattice for key " << key;

      if (duration_model_scale != 0.0) {
        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / duration_model_scale),
                          &clat);
        ArcSort(&clat, fst::OLabelCompare<CompactLatticeArc>());

        // Insert the phone-id/duration info into the lattice olabels
        DurationModelReplaceLabelsLattice(&clat,
                                          trans_model,
                                          trans_model.NumPhones());

        // Wrap the duration-model scorer with an on-demand fst
        PhoneDurationModelDeterministicFst on_demand_fst(
                                               trans_model.NumPhones(),
                                               nnet_durmodel.GetDurationModel(),
                                               &durmodel_scorer);

        // Compose the lattice with the on-demand fst
        CompactLattice composed_clat;
        ComposeCompactLatticeDeterministic(clat,
                                           &on_demand_fst,
                                           &composed_clat);
        // Replace the labels back
        DurationModelReplaceLabelsBackLattice(&composed_clat);

        // Determinizes the composed lattice.
        Lattice composed_lat;
        ConvertLattice(composed_clat, &composed_lat);
        Invert(&composed_lat);
        CompactLattice determinized_clat;
        DeterminizeLattice(composed_lat, &determinized_clat);
        fst::ScaleLattice(fst::GraphLatticeScale(duration_model_scale),
                          &determinized_clat);
        if (determinized_clat.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty lattice for utterance " << key;
          n_fail++;
        } else {
          compact_lattice_writer.Write(key, determinized_clat);
          n_done++;
        }
      } else {
        // Zero scale so nothing to do.
        n_done++;
        compact_lattice_writer.Write(key, clat);
      }
    }

    KALDI_LOG << "Rescored " << n_done << " lattices with " << n_fail
      << " failures.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


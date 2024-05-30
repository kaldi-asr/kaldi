// sgmm2bin/sgmm2-est.cc

// Copyright 2009-2012  Saarland University (Author:  Arnab Ghoshal)
//                      Johns Hopkins University (Author: Daniel Povey)

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
#include "util/kaldi-thread.h"
#include "sgmm2/am-sgmm2.h"
#include "hmm/transition-model.h"
#include "sgmm2/estimate-am-sgmm2.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Estimate SGMM model parameters from accumulated stats.\n"
        "Usage: sgmm2-est [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = true;
    std::string update_flags_str = "vMNwucSt";
    std::string write_flags_str = "gsnu";
    kaldi::MleTransitionUpdateConfig tcfg;
    kaldi::MleAmSgmm2Options sgmm_opts;
    kaldi::Sgmm2SplitSubstatesConfig split_opts;
    int32 increase_phn_dim = 0;
    int32 increase_spk_dim = 0;
    bool remove_speaker_space = false;
    bool spk_dep_weights = false;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("increase-phn-dim", &increase_phn_dim, "Increase phone-space "
                "dimension as far as allowed towards this target.");
    po.Register("increase-spk-dim", &increase_spk_dim, "Increase speaker-space "
                "dimension as far as allowed towards this target.");
    po.Register("spk-dep-weights", &spk_dep_weights, "If true, have speaker-"
                "dependent weights (symmetric SGMM)-- this option only makes "
                "a difference if you use the --increase-spk-dim option and "
                "are increasing the speaker dimension from zero.");
    po.Register("remove-speaker-space", &remove_speaker_space, "Remove speaker-specific "
                "projections N");
    po.Register("write-occs", &occs_out_filename, "File to write pdf "
                "occupation counts to.");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vMNwcSt.");
    po.Register("write-flags", &write_flags_str, "Which SGMM parameters to "
                "write: subset of gsnu");
    po.Register("num-threads", &g_num_threads, "Number of threads to use in "
                "weight update and normalizer computation");
    tcfg.Register(&po);
    sgmm_opts.Register(&po);
    split_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    kaldi::SgmmUpdateFlagsType update_flags =
        StringToSgmmUpdateFlags(update_flags_str);
    kaldi::SgmmWriteFlagsType write_flags =
        StringToSgmmWriteFlags(write_flags_str);
    
    AmSgmm2 am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    MleAmSgmm2Accs sgmm_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      sgmm_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    if (update_flags & kSgmmTransitions) {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: Overall " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    sgmm_accs.Check(am_sgmm, true); // Will check consistency and print some diagnostics.

    { // Do the update.
      kaldi::MleAmSgmm2Updater updater(sgmm_opts);
      updater.Update(sgmm_accs, &am_sgmm, update_flags);
    }

    Vector<BaseFloat> pdf_occs;
    sgmm_accs.GetStateOccupancies(&pdf_occs);

    if (split_opts.split_substates != 0)
      am_sgmm.SplitSubstates(pdf_occs, split_opts);

    if (!occs_out_filename.empty()) {
      kaldi::Output ko(occs_out_filename, binary_write);
      pdf_occs.Write(ko.Stream(), binary_write);
    }

    if (increase_phn_dim != 0 || increase_spk_dim != 0) {
      // Feature normalizing transform matrix used to initialize the new columns
      // of the phonetic- or speaker-space projection matrices.
      kaldi::Matrix<BaseFloat> norm_xform;
      ComputeFeatureNormalizingTransform(am_sgmm.full_ubm(), &norm_xform);
      if (increase_phn_dim != 0)
        am_sgmm.IncreasePhoneSpaceDim(increase_phn_dim, norm_xform);
      if (increase_spk_dim != 0)
        am_sgmm.IncreaseSpkSpaceDim(increase_spk_dim, norm_xform,
                                    spk_dep_weights);
    }
    if (remove_speaker_space) {
      KALDI_LOG << "Removing speaker space (projections N_)";
      am_sgmm.RemoveSpeakerSpace();
    }

    am_sgmm.ComputeDerivedVars(); // recompute normalizers, and possibly
    // weights.
    
    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_sgmm.Write(ko.Stream(), binary_write, write_flags);
    }
    
    
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



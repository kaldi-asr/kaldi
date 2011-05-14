// sgmmbin/sgmm-est.cc

// Copyright 2009-2011  Arnab Ghoshal

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

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "sgmm/estimate-am-sgmm.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Estimate SGMM model parameters from accumulated stats.\n"
        "Usage: sgmm-estimate [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = false;
    std::string update_flags_str = "vMNwcS";
    kaldi::TransitionUpdateConfig tcfg;
    kaldi::MleAmSgmmOptions sgmm_opts;
    int32 split_substates = 0;
    int32 increase_phn_dim = 0;
    int32 increase_spk_dim = 0;
    bool remove_speaker_space = false;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat max_cond = 100;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("split-substates", &split_substates, "Increase number of "
        "substates to this overall target.");
    po.Register("increase-phn-dim", &increase_phn_dim, "Increase phone-space "
        "dimension to this overall target.");
    po.Register("increase-spk-dim", &increase_spk_dim, "Increase speaker-space "
        "dimension to this overall target.");
    po.Register("remove-speaker-space", &remove_speaker_space, "Remove speaker-specific "
                "projections N");
    po.Register("power", &power, "Exponent for substate occupancies used while"
        "splitting substates.");
    po.Register("perturb-factor", &perturb_factor, "Perturbation factor for "
        "state vectors while splitting substates.");
    po.Register("max-cond-split", &max_cond, "Max condition number of smoothing "
        "matrix used in substate splitting.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vMNwcS.");
    tcfg.Register(&po);
    sgmm_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    kaldi::SgmmUpdateFlagsType acc_flags = StringToSgmmUpdateFlags(update_flags_str);

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    Vector<double> transition_accs;
    MleAmSgmmAccs sgmm_accs;
    {
      bool binary;
      Input is(stats_filename, &binary);
      transition_accs.Read(is.Stream(), binary);
      sgmm_accs.Read(is.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.Update(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    {  // Update SGMM.
//      BaseFloat objf_impr, count;
      kaldi::MleAmSgmmUpdater sgmm_updater(sgmm_opts);
      sgmm_updater.Update(sgmm_accs, &am_sgmm, acc_flags);
//      KALDI_LOG << "GMM update: average " << (objf_impr/count)
//                << " objective function improvement per frame over "
//                <<  (count) <<  " frames.";
    }

    if (split_substates != 0 || !occs_out_filename.empty()) {  // get state occs
      Vector<BaseFloat> state_occs;
      sgmm_accs.GetStateOccupancies(&state_occs);

      if (split_substates != 0) {
        am_sgmm.SplitSubstates(state_occs, split_substates, perturb_factor,
                               power, max_cond);
        am_sgmm.ComputeDerivedVars();  // recompute normalizers...
      }

      if (!occs_out_filename.empty()) {
        kaldi::Output os(occs_out_filename, binary_write);
        state_occs.Write(os.Stream(), binary_write);
      }
    }

    if (increase_phn_dim != 0 || increase_spk_dim != 0) {
      // Feature normalizing transform matrix used to initialize the new columns
      // of the phonetic- or speaker-space projection matrices.
      kaldi::Matrix<BaseFloat> norm_xform;
      ComputeFeatureNormalizer(am_sgmm.full_ubm(), &norm_xform);
      if (increase_phn_dim != 0)
        am_sgmm.IncreasePhoneSpaceDim(increase_phn_dim, norm_xform);
      if (increase_spk_dim != 0)
        am_sgmm.IncreaseSpkSpaceDim(increase_spk_dim, norm_xform);
    }
    if (remove_speaker_space) {
      KALDI_LOG << "Removing speaker space (projections N_)";
      am_sgmm.RemoveSpeakerSpace();
    }

    {
      Output os(model_out_filename, binary_write);
      trans_model.Write(os.Stream(), binary_write);
      am_sgmm.Write(os.Stream(), binary_write, kSgmmWriteAll);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}



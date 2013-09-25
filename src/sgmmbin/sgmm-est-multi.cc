// sgmmbin/sgmm-est-multi.cc

// Copyright 2009-2012  Arnab Ghoshal

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

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "sgmm/estimate-am-sgmm.h"
#include "sgmm/estimate-am-sgmm-multi.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  // Memory for these will be freed in the catch block in case of exceptions.
  std::vector<AmSgmm*> sgmms_in;
  std::vector<MleAmSgmmAccs*> sgmm_accs_in;
  std::vector<TransitionModel*> trans_models_in;

  try {
    typedef kaldi::int32 int32;
    const char *usage =
        "Estimate multiple SGMM models from corresponding stats, such that the"
        " global parameters\n(phone-, speaker-, and weight-projections and "
        "covariances) are tied across models.\n"
        "Usage: sgmm-est-multi [options] <model1> <stats1> <model1_out> <occs1_out> [<model2> "
        "<stats2> <model2_out> <occs2_out> ...]\n";

    bool binary_write = true;
    std::string update_flags_str = "vMNwcSt";
    std::string write_flags_str = "gsnu";
    kaldi::MleTransitionUpdateConfig tcfg;
    kaldi::MleAmSgmmOptions sgmm_opts;
    std::string split_substates = "";  // Space-seperated list of #substates
    std::vector<int32> split_substates_int;  // The above string split on space
    int32 increase_phn_dim = 0;
    int32 increase_spk_dim = 0;
    bool remove_speaker_space = false;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat max_cond = 100;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    // The split-substates option also takes a single integer: the same number
    // of substates for all models.
    po.Register("split-substates", &split_substates, "Space-separated string "
                "with target number of substates for each model.");
    po.Register("increase-phn-dim", &increase_phn_dim, "Increase phone-space "
                "dimension as far as allowed towards this target.");
    po.Register("increase-spk-dim", &increase_spk_dim, "Increase speaker-space "
                "dimension as far as allowed towards this target.");
    po.Register("remove-speaker-space", &remove_speaker_space,
                "Remove speaker-specific projections N");
    po.Register("power", &power, "Exponent for substate occupancies used while "
                "splitting substates.");
    po.Register("perturb-factor", &perturb_factor, "Perturbation factor for "
                "state vectors while splitting substates.");
    po.Register("max-cond-split", &max_cond, "Max condition number of smoothing "
                "matrix used in substate splitting.");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vMNwcSt.");
    po.Register("write-flags", &write_flags_str, "Which SGMM parameters to "
                "write: subset of gsnu");
    tcfg.Register(&po);
    sgmm_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() <= 0 || (po.NumArgs() % 4 != 0)) {
      po.PrintUsage();
      exit(1);
    }
    // How many 4-tuples of model, stats, output model, output occs
    int32 num_models = po.NumArgs()/4;
    sgmms_in.resize(num_models, NULL);
    sgmm_accs_in.resize(num_models, NULL);
    trans_models_in.resize(num_models, NULL);

    if (!split_substates.empty()) {
      SplitStringToIntegers(split_substates, " ", true /*omit empty strings*/,
                            &split_substates_int);
      if (split_substates_int.size() == 1) {  // Same #substates for all models
        int32 tmp_int = split_substates_int[0];
        split_substates_int.resize(num_models, tmp_int);
      }
      if (split_substates_int.size() != num_models) {
        KALDI_ERR << "Found " << split_substates_int.size() << " splitting "
                  << "targets; expecting 1 or " << num_models;
      }
    }

    SgmmUpdateFlagsType update_flags = StringToSgmmUpdateFlags(update_flags_str);
    SgmmWriteFlagsType write_flags = StringToSgmmWriteFlags(write_flags_str);

    std::vector<std::string> model_out_filenames(num_models);
    std::vector<std::string> occs_out_filenames(num_models);
    int32 phn_dim, spk_dim, num_gauss, feat_dim;

    for (int i = 0; i < num_models; ++i) {
      std::string model_in_filename = po.GetArg(i*4+1),
          stats_filename = po.GetArg(i*4+2);
      model_out_filenames[i] = po.GetArg(i*4+3);
      occs_out_filenames[i] = po.GetArg(i*4+4);

      AmSgmm *am_sgmm = new AmSgmm();
      TransitionModel *trans_model = new TransitionModel();
      {
        bool binary;
        Input ki(model_in_filename, &binary);
        trans_model->Read(ki.Stream(), binary);
        am_sgmm->Read(ki.Stream(), binary);
      }
      if (i == 0) {
        phn_dim = am_sgmm->PhoneSpaceDim();
        spk_dim = am_sgmm->SpkSpaceDim();
        num_gauss = am_sgmm->NumGauss();
        feat_dim = am_sgmm->FeatureDim();
      } else {
        if (am_sgmm->PhoneSpaceDim() != phn_dim) {
          KALDI_ERR << "File '" << model_in_filename << "': mismatched "
                    << "phone-space dim: expecting " << phn_dim << ", found "
                    << am_sgmm->PhoneSpaceDim();
        }
        if (am_sgmm->SpkSpaceDim() != spk_dim) {
          KALDI_ERR << "File '" << model_in_filename << "': mismatched "
                    << "speaker-space dim: expecting " << spk_dim << ", found "
                    << am_sgmm->SpkSpaceDim();
        }
        if (am_sgmm->NumGauss() != num_gauss) {
          KALDI_ERR << "File '" << model_in_filename << "': mismatched UBM "
                    << "size: expecting " << num_gauss << ", found "
                    << am_sgmm->NumGauss();
        }
        if (am_sgmm->FeatureDim() != feat_dim) {
          KALDI_ERR << "File '" << model_in_filename << "': mismatched feature "
                    << "dim: expecting " << feat_dim << ", found "
                    << am_sgmm->FeatureDim();
        }
      }
      sgmms_in[i] = am_sgmm;
      trans_models_in[i] = trans_model;

      Vector<double> transition_accs;
      MleAmSgmmAccs *sgmm_accs = new MleAmSgmmAccs();
      {
        bool binary;
        Input ki(stats_filename, &binary);
        transition_accs.Read(ki.Stream(), binary);
        sgmm_accs->Read(ki.Stream(), binary, false);
      }
      // Check consistency and print some diagnostics.
      sgmm_accs->Check(*am_sgmm, true);
      sgmm_accs_in[i] = sgmm_accs;

      if (update_flags & kSgmmTransitions) {  // Update transition model.
        BaseFloat objf_impr, count;
        KALDI_LOG << "Updating transitions for model: " << model_in_filename;
        trans_model->MleUpdate(transition_accs, tcfg, &objf_impr, &count);
        KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                  << " log-like improvement per frame over " << (count)
                  << " frames";
      }
    }

    {  // Update all the SGMMs together.
      kaldi::MleAmSgmmUpdaterMulti multi_sgmm_updater(*sgmms_in[0], sgmm_opts);
      multi_sgmm_updater.Update(sgmm_accs_in, sgmms_in, update_flags);
    }

    for (int i = 0; i < num_models; ++i) {
      Vector<BaseFloat> state_occs;
      sgmm_accs_in[i]->GetStateOccupancies(&state_occs);

      if (!split_substates.empty()) {
        sgmms_in[i]->SplitSubstates(state_occs, split_substates_int[i], perturb_factor,
                                    power, max_cond);
        sgmms_in[i]->ComputeDerivedVars();  // recompute normalizers...
      }

      {
        kaldi::Output ko(occs_out_filenames[i], false /* no binary write */);
        state_occs.Write(ko.Stream(), false /* no binary write */);
      }

      if (increase_phn_dim != 0 || increase_spk_dim != 0) {
        // Feature normalizing transform matrix used to initialize the new columns
        // of the phonetic- or speaker-space projection matrices.
        kaldi::Matrix<BaseFloat> norm_xform;
        ComputeFeatureNormalizer(sgmms_in[i]->full_ubm(), &norm_xform);
        if (increase_phn_dim != 0)
          sgmms_in[i]->IncreasePhoneSpaceDim(increase_phn_dim, norm_xform);
        if (increase_spk_dim != 0)
          sgmms_in[i]->IncreaseSpkSpaceDim(increase_spk_dim, norm_xform);
      }
      if (remove_speaker_space) {
        KALDI_LOG << "Removing speaker space (projections N_)";
        sgmms_in[i]->RemoveSpeakerSpace();
      }

      {
        Output ko(model_out_filenames[i], binary_write);
        trans_models_in[i]->Write(ko.Stream(), binary_write);
        sgmms_in[i]->Write(ko.Stream(), binary_write, write_flags);
        KALDI_LOG << "Written model to " << model_out_filenames[i];
      }
    }
    return 0;
  } catch(const std::exception& e) {
    kaldi::DeletePointers(&sgmms_in);
    kaldi::DeletePointers(&sgmm_accs_in);
    kaldi::DeletePointers(&trans_models_in);
    std::cerr << e.what();
    return -1;
  }
}



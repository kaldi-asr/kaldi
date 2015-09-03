// sgmmbin/sgmm-mixup.cc

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Increase number of sub-states or dimensions in SGMM\n"
        "Usage: sgmm-mixup [options] <model-in> <model-out>\n"
        "E.g. of mixing up:\n"
        " sgmm-mixup --read-occs=1.occs --num-substates=10000 1.mdl 2.mdl\n"
        "E.g. of increasing phonetic dim:\n"
        " sgmm-mixup --increase-phn-dim=50 1.mdl 2.mdl\n"
        "E.g. of increasing speaker dim:\n"
        " sgmm-mixup --increase-spk-dim=50 1.mdl 2.mdl\n"
        "E.g. of removing speaker space:\n"
        " sgmm-mixup --remove-speaker-space 1.mdl 2.mdl\n"
        "These modes may be combined.\n";
    
    bool binary_write = true;
    std::string write_flags_str = "gsnu";
    int32 split_substates = 0;
    int32 increase_phn_dim = 0;
    int32 increase_spk_dim = 0;
    bool remove_speaker_space = false;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat max_cond = 100;
    std::string occs_in_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("split-substates", &split_substates, "Increase number of "
                "substates to this overall target.");
    po.Register("increase-phn-dim", &increase_phn_dim, "Increase phone-space "
                "dimension as far as allowed towards this target.");
    po.Register("increase-spk-dim", &increase_spk_dim, "Increase speaker-space "
                "dimension as far as allowed towards this target.");
    po.Register("remove-speaker-space", &remove_speaker_space, "Remove speaker-specific "
                "projections N");
    po.Register("power", &power, "Exponent for substate occupancies used while "
                "splitting substates.");
    po.Register("perturb-factor", &perturb_factor, "Perturbation factor for "
                "state vectors while splitting substates.");
    po.Register("max-cond-split", &max_cond, "Max condition number of smoothing "
                "matrix used in substate splitting.");
    po.Register("write-flags", &write_flags_str, "Which SGMM parameters to "
                "write: subset of gsnu");
    po.Register("read-occs", &occs_in_filename, "Read occupancies from this file "
                "(required for mixing up)");
    
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    kaldi::SgmmWriteFlagsType write_flags =
        StringToSgmmWriteFlags(write_flags_str);
    
    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    if (split_substates != 0) {
      if (occs_in_filename.empty())
        KALDI_ERR << "The --split-substates option requires the --read-occs option";
      
      Vector<BaseFloat> state_occs;
      {
        bool binary_in;
        kaldi::Input ki(occs_in_filename, &binary_in);
        state_occs.Read(ki.Stream(), binary_in);
      }
      
      am_sgmm.SplitSubstates(state_occs, split_substates, perturb_factor,
                             power, max_cond);
      am_sgmm.ComputeDerivedVars();  // recompute normalizers...
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



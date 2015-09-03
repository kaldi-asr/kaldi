// gmmbin/gmm-est-map.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    
    const char *usage =
        "Do Maximum A Posteriori re-estimation of GMM-based acoustic model\n"
        "Usage:  gmm-est-map [options] <model-in> <stats-in> <model-out>\n"
        "e.g.: gmm-est-map 1.mdl 1.acc 2.mdl\n";

    bool binary_write = true;
    MapTransitionUpdateConfig tcfg;
    MapDiagGmmOptions gmm_opts;
    std::string update_flags_str = "mvwt";
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
                "update: subset of mvwt.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    tcfg.Register(&po);
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    kaldi::GmmFlagsType update_flags =
        StringToGmmFlags(update_flags_str);

    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    Vector<double> transition_accs;
    AccumAmDiagGmm gmm_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      gmm_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    if (update_flags & kGmmTransitions) {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.MapUpdate(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: Overall " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    {  // Update GMMs.
      BaseFloat objf_impr, count;
      BaseFloat tot_like = gmm_accs.TotLogLike(),
          tot_t = gmm_accs.TotCount();
      MapAmDiagGmmUpdate(gmm_opts, gmm_accs, update_flags, &am_gmm,
                         &objf_impr, &count);
      KALDI_LOG << "GMM update: Overall " << (objf_impr/count)
                << " objective function improvement per frame over "
                <<  count <<  " frames";
      KALDI_LOG << "GMM update: Overall avg like per frame = "
                << (tot_like/tot_t) << " over " << tot_t << " frames.";
    }

    if (!occs_out_filename.empty()) {  // get state occs
      Vector<BaseFloat> state_occs;
      state_occs.Resize(gmm_accs.NumAccs());
      for (int i = 0; i < gmm_accs.NumAccs(); i++)
        state_occs(i) = gmm_accs.GetAcc(i).occupancy().Sum();
      bool binary = false;
      WriteKaldiObject(state_occs, occs_out_filename, binary);
    }
    
    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



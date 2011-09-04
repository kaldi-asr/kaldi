// tiedbin/tied-full-gmm-est.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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
#include "tied/am-tied-full-gmm.h"
#include "tied/mle-am-tied-full-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    MleFullGmmOptions gmm_opts;
    MleTiedGmmOptions tied_opts;

    const char *usage =
        "Accumulate stats for GMM training.\n"
        "Usage:  tied-full-gmm-est [options] <model-in> <stats-in> <model-out>\n"
        "e.g.: tied-full-gmm-est 1.mdl 1.acc 2.mdl\n";

    bool binary_write = false;
    TransitionUpdateConfig tcfg;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("write-occs", &occs_out_filename, "File to write state "
        "occupancies to.");
    tcfg.Register(&po);
    gmm_opts.Register(&po);
    tied_opts.Register(&po);

    po.Read(argc, argv);

    if (gmm_opts.remove_low_count_gaussians) {
      KALDI_WARN << "enforcing gmm_opts.remove_low_count_gaussians = false";
      gmm_opts.remove_low_count_gaussians = false;
    }

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    AmTiedFullGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input is(model_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
      am_gmm.Read(is.Stream(), binary_read);
    }

    Vector<double> transition_accs;
    AccumAmTiedFullGmm gmm_accs;
    {
      bool binary;
      Input is(stats_filename, &binary);
      transition_accs.Read(is.Stream(), binary);
      gmm_accs.Read(is.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.Update(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    {  // Update GMMs.
      BaseFloat objf_impr, count;
      MleAmTiedFullGmmUpdate(gmm_opts, tied_opts, gmm_accs, kGmmAll, &am_gmm, &objf_impr, &count);
      KALDI_LOG << "GMM update: average " << (objf_impr/count)
                << " objective function improvement per frame over "
                <<  (count) <<  " frames.";
    }

    if (!occs_out_filename.empty()) {  // get state occs (only for tied gmms)
      Vector<BaseFloat> state_occs;
      state_occs.Resize(gmm_accs.NumTiedAccs());
      for (int i = 0; i < gmm_accs.NumTiedAccs(); i++)
        state_occs(i) = gmm_accs.GetTiedAcc(i).occupancy().Sum();

      if (!occs_out_filename.empty()) {
        kaldi::Output ko(occs_out_filename, binary_write);
        state_occs.Write(ko.Stream(), binary_write);
      }
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



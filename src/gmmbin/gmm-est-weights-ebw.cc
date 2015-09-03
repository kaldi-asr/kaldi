// gmmbin/gmm-est-weights-ebw.cc

// Copyright 2009-2011  Petr Motlicek  Chao Weng

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
#include "gmm/ebw-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Do EBW update on weights for MMI, MPE or MCE discriminative training.\n"
        "Numerator stats should not be I-smoothed\n"
        "Usage:  gmm-est-weights-ebw [options] <model-in> <stats-num-in> <stats-den-in> <model-out>\n"
        "e.g.: gmm-est-weights-ebw 1.mdl num.acc den.acc 2.mdl\n";

    bool binary_write = false;
    std::string update_flags_str = "w";

    EbwWeightOptions ebw_weight_opts;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
                "update; only \"w\" flag is looked at.");
    
    ebw_weight_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    kaldi::GmmFlagsType update_flags =
        StringToGmmFlags(update_flags_str);    

    std::string model_in_filename = po.GetArg(1),
        num_stats_filename = po.GetArg(2),
        den_stats_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    Vector<double> num_transition_accs; // won't be used.
    Vector<double> den_transition_accs; // won't be used.

    AccumAmDiagGmm num_stats;
    AccumAmDiagGmm den_stats;
    {
      bool binary;
      Input ki(num_stats_filename, &binary);
      num_transition_accs.Read(ki.Stream(), binary);
      num_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }
    
    {
      bool binary;
      Input ki(den_stats_filename, &binary);
      num_transition_accs.Read(ki.Stream(), binary);
      den_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }
    
    if (update_flags & kGmmWeights) {  // Update weights.
      BaseFloat auxf_impr, count;
      UpdateEbwWeightsAmDiagGmm(num_stats, den_stats, ebw_weight_opts, &am_gmm,
                                &auxf_impr, &count);
      KALDI_LOG << "Num count " << num_stats.TotCount() << ", den count "
                << den_stats.TotCount();
      KALDI_LOG << "Overall auxf impr/frame from weight update is " << (auxf_impr/count)
                << " over " << count << " frames.";
    } else {
      KALDI_LOG << "Doing nothing because flags do not specify to update the weights.";
    }
      
    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

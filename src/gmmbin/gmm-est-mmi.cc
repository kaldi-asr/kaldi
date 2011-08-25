// gmmbin/gmm-est-mmi.cc

// Copyright 2009-2011  Petr Motlicek

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
#include "gmm/mmie-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    MmieDiagGmmOptions mmi_opts;

    const char *usage =
        "Accumulate stats for MMI discriminative training.\n"
        "Usage:  gmm-est-mmi [options] <model-in> <stats-num-in> <stats-den-in> <model-out>\n"
        "e.g.: gmm-est 1.mdl num.acc den.acc 2.mdl\n";

    bool binary_write = false;
    //TransitionUpdateConfig tcfg;
    BaseFloat ebwe = 2.0;
    BaseFloat tau = 100.0;
    std::string occs_out_filename;


    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode"
                "target.");
    po.Register("tau", &tau, "Tau i-smoothing"
        " constant.");
    po.Register("ebw_e", &ebwe, "Ebw_e "
        "constant.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
        "occupancies to.");
    //tcfg.Register(&po);
    mmi_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string model_in_filename = po.GetArg(1),
        num_stats_filename = po.GetArg(2),
        den_stats_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4);


    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input is(model_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
      am_gmm.Read(is.Stream(), binary_read);
    }

    Vector<double> num_transition_accs;
    Vector<double> den_transition_accs;
    MmieAccumAmDiagGmm mmi_accs;
    {
      bool binary;
      Input is(num_stats_filename, &binary);
      num_transition_accs.Read(is.Stream(), binary);
      mmi_accs.ReadNum(is.Stream(), binary, true);  // true == add; doesn't matter here.
    }
    {
      bool binary;
      Input is(den_stats_filename, &binary);
      num_transition_accs.Read(is.Stream(), binary);
      mmi_accs.ReadDen(is.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    

   
    {  // Update GMMs.
      BaseFloat objf_impr, count;
      MmieAmDiagGmmUpdate(mmi_opts, mmi_accs, kGmmAll, &am_gmm, &objf_impr, &count);
      KALDI_LOG << "GMM update: average " << (objf_impr/count)
                << " objective function improvement per frame over "
                <<  (count) <<  " frames.";
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



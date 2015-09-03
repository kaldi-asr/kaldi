// gmmbin/gmm-global-est.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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
#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    MleDiagGmmOptions gmm_opts;

    const char *usage =
        "Estimate a diagonal-covariance GMM from the accumulated stats.\n"
        "Usage:  gmm-global-est [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = true;
    int32 mixup = 0;
    BaseFloat perturb_factor = 0.01;
    std::string update_flags_str = "mvw"; 
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters will be "
                "updated: subset of mvw.");
    po.Register("mix-up", &mixup, "Increase number of mixture components to "
                "this overall target.");
    po.Register("perturb-factor", &perturb_factor, "While mixing up, perturb "
        "means by standard deviation times this factor.");
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    DiagGmm gmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      gmm.Read(ki.Stream(), binary_read);
    }

    AccumDiagGmm gmm_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      gmm_accs.Read(ki.Stream(), binary, true /* add accs, doesn't matter */);
    }

    {  // Update GMMs.
      BaseFloat objf_impr, count;
      MleDiagGmmUpdate(gmm_opts, gmm_accs,
                       StringToGmmFlags(update_flags_str),
                       &gmm, &objf_impr, &count);
      KALDI_LOG << "Overall objective function improvement is "
                << (objf_impr/count) << " per frame over "
                <<  (count) <<  " frames.";
    }

    if (mixup != 0)
      gmm.Split(mixup, perturb_factor);

    WriteKaldiObject(gmm, model_out_filename, binary_write);

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



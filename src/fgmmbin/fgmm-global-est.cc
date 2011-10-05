// fgmmbin/fgmm-global-est.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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
#include "gmm/full-gmm.h"
#include "gmm/estimate-full-gmm.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    kaldi::MleFullGmmOptions gmm_opts;

    const char *usage =
        "Estimate a full-covariance GMM from the accumulated stats.\n"
        "Usage:  fgmm-global-est [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = true;
    int32 mixup = 0;
    kaldi::BaseFloat perturb_factor = 0.01;

    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
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

    kaldi::FullGmm fgmm;
    {
      bool binary_read;
      kaldi::Input is(model_in_filename, &binary_read);
      fgmm.Read(is.Stream(), binary_read);
    }

    kaldi::MlEstimateFullGmm gmm_accs;
    {
      bool binary;
      kaldi::Input is(stats_filename, &binary);
      gmm_accs.Read(is.Stream(), binary, true /* add accs, doesn't matter */);
    }

    {  // Update GMMs.
      kaldi::BaseFloat objf_impr, count;
      gmm_accs.Update(gmm_opts, kaldi::kGmmAll, &fgmm, &objf_impr, &count);
      KALDI_LOG << "GMM update: average " << (objf_impr/count)
                << " objective function improvement per frame over "
                <<  (count) <<  " frames.";
    }

    if (mixup != 0)
      fgmm.Split(mixup, perturb_factor);

    {
      kaldi::Output os(model_out_filename, binary_write);
      fgmm.Write(os.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



// gmmbin/gmm-mixup.cc

// Copyright 2009-2011  Microsoft Corporation

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
        "Does GMM mixing up (and Gaussian merging)\n"
        "Usage:  gmm-mixup [options] <model-in> <state-occs-in> <model-out>\n"
        "e.g. of mixing up:\n"
        " gmm-mixup --mix-up=4000 1.mdl 1.occs 2.mdl\n"
        "e.g. of merging:\n"
        " gmm-mixup --merge=2000 1.mdl 1.occs 2.mdl\n";
        
    bool binary_write = true;
    int32 mixup = 0;
    int32 mixdown = 0;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat min_count = 20.0;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("mix-up", &mixup, "Increase number of mixture components to "
                "this overall target.");
    po.Register("min-count", &min_count,
                "Minimum count enforced while mixing up.");
    po.Register("mix-down", &mixdown, "If nonzero, merge mixture components to this "
                "target.");
    po.Register("power", &power, "If mixing up, power to allocate Gaussians to"
        " states.");
    po.Register("perturb-factor", &perturb_factor, "While mixing up, perturb "
        "means by standard deviation times this factor.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string model_in_filename = po.GetArg(1),
        occs_in_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    if (mixup != 0 || mixdown != 0) {

      Vector<BaseFloat> occs;
      ReadKaldiObject(occs_in_filename, &occs);
      if (occs.Dim() != am_gmm.NumPdfs())
        KALDI_ERR << "Dimension of state occupancies " << occs.Dim()
                   << " does not match num-pdfs " << am_gmm.NumPdfs();

      if (mixdown != 0)
        am_gmm.MergeByCount(occs, mixdown, power, min_count);

      if (mixup != 0)
        am_gmm.SplitByCount(occs, mixup, perturb_factor,
                            power, min_count);
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



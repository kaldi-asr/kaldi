// gmmbin/fmpe-gmm-est.cc

// Copyright 2009-2011  Yanmin Qian

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
#include "gmm/fmpe-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Estimate fMPE transforms.\n"
        "Note: not yet tested.\n"
        "Usage:  fmpe-gmm-est [options] <am-model-in> <fmpe-proj-matrix-in> <stats-in> <fmpe-proj-matrix-out>\n"
        "e.g.: gmm-est 1.mdl 1.mat 1.acc 2.mat\n";

    bool binary_write = false;
    FmpeConfig fmpe_opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    fmpe_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }


    std::string model_in_filename = po.GetArg(1),
        fmpe_proj_mat_in_filename = po.GetArg(2),
        stats_filename = po.GetArg(3),
        fmpe_proj_mat_out_filename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    FmpeAccs fmpe_accs(fmpe_opts);
    {
      bool binary;
      Input ki(stats_filename, &binary);
      fmpe_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    FmpeUpdater fmpe_updater(fmpe_accs);
    {
      bool binary;
      Input ki(fmpe_proj_mat_in_filename, &binary);
      fmpe_updater.Read(ki.Stream(), binary);
    }

    {  // update the Fmpe projection matrix
      BaseFloat obj_change_out, count_out;
      fmpe_updater.ComputeAvgStandardDeviation(am_gmm);
      fmpe_updater.Update(fmpe_accs, &obj_change_out, &count_out);
    }

    {
      Output ko(fmpe_proj_mat_out_filename, binary_write);
      fmpe_updater.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written Fmpe projection matrix to " << fmpe_proj_mat_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



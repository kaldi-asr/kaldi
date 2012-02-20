// gmmbin/fmpe-gmm-model-diffs-est.cc

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
#include "gmm/mle-am-diag-gmm.h"
//#include "gmm/ebw-am-diag-gmm.h"   // TODO wait Arnab to finish the AccumAmEbwDiagGmm Class, then make it active
#include "gmm/fmpe-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Compute the model parameters differentials from the ebw accumulators (in mpe training) for fmpe training.\n"
        "Usage:  fmpe-gmm-model-diffs-est [options] <model-in> <ebw-stats-in> <mle-stats-in> <model-diffs-out>\n"
        "e.g.: fmpe-gmm-model-diff-est 1.mdl 1.ebw.acc 1.mle.acc 1.model.diffs\n";

    bool binary = false;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }


    std::string model_in_filename = po.GetArg(1),
        ebw_stats_in_filename = po.GetArg(2),
        mle_stats_in_filename = po.GetArg(3),
        model_diffs_out_filename = po.GetArg(4);


    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    Vector<double> transition_ebw_accs;
//    AccumAmEbwDiagGmm gmm_ebw_accs;  // TODO wait Arnab to finish the AccumAmEbwDiagGmm Class, then make it active
    {
      bool binary;
      Input ki(ebw_stats_in_filename, &binary);
      transition_ebw_accs.Read(ki.Stream(), binary);
      // TODO wait Arnab to finish the AccumAmEbwDiagGmm Class, then make it active
 //     gmm_ebw_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    Vector<double> transition_mle_accs;
    AccumAmDiagGmm gmm_mle_accs;
    {
      bool binary;
      Input ki(mle_stats_in_filename, &binary);
      transition_mle_accs.Read(ki.Stream(), binary);
      gmm_mle_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    std::vector<FmpeAccumModelDiff*> model_diffs;
    model_diffs.reserve(am_gmm.NumPdfs());
    for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
      model_diffs.push_back(new FmpeAccumModelDiff(am_gmm.GetPdf(i)));
      // TODO wait Arnab to finish the AccumAmEbwDiagGmm Class, then make it active
//      model_diff.back()->ComputeModelParaDiff(am_gmm.GetPdf(i), gmm_ebw_acc.GetAcc(i), gmm_mle_accs.GetAcc(i));
    }

    // Write out the model diffs
    {
      kaldi::Output ko(model_diffs_out_filename, binary);
      WriteToken(ko.Stream(), binary, "<DIMENSION>");
      WriteBasicType(ko.Stream(), binary, static_cast<int32>(am_gmm.Dim()));
      WriteToken(ko.Stream(), binary, "<NUMPDFS>");
      WriteBasicType(ko.Stream(), binary, static_cast<int32>(model_diffs.size()));
      for (std::vector<FmpeAccumModelDiff*>::const_iterator it = model_diffs.begin(),
        end = model_diffs.end(); it != end; ++it) {
        (*it)->Write(ko.Stream(), binary);
      }
    }

    KALDI_LOG << "Written model diffs to " << model_diffs_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



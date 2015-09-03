// sgmm2bin/sgmm2-est-ebw.cc

// Copyright 2012  Johns Hopkins Univerity (Author: Daniel Povey)

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
#include "thread/kaldi-thread.h"
#include "hmm/transition-model.h"
#include "sgmm2/estimate-am-sgmm2-ebw.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  using std::string;
  try {
    const char *usage =
        "Estimate SGMM model parameters discriminatively using Extended\n"
        "Baum-Welch style of update\n"
        "Usage: sgmm2-est-ebw [options] <model-in> <num-stats-in> <den-stats-in> <model-out>\n";


    string update_flags_str = "vMNwcSt";
    bool binary_write = true;
    string write_flags_str = "gsnu";
    EbwAmSgmm2Options opts;

    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vMNwcSt.");
    po.Register("write-flags", &write_flags_str, "Which SGMM parameters to "
                "write: subset of gsnu");
    po.Register("num-threads", &g_num_threads, "Number of threads to use in "
                "weight update and normalizer computation");
    opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    string model_in_filename = po.GetArg(1),
        num_stats_filename = po.GetArg(2),
        den_stats_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4);
    
    SgmmUpdateFlagsType update_flags = StringToSgmmUpdateFlags(update_flags_str);
    SgmmWriteFlagsType write_flags = StringToSgmmWriteFlags(write_flags_str);

    AmSgmm2 am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    MleAmSgmm2Accs sgmm_num_accs;
    {
      bool binary;
      Vector<double> transition_accs; // won't be used.
      Input ki(num_stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      sgmm_num_accs.Read(ki.Stream(), binary, false);  // false == add; doesn't matter.
    }
    MleAmSgmm2Accs sgmm_den_accs;
    {
      bool binary;
      Vector<double> transition_accs; // won't be used.
      Input ki(den_stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      sgmm_den_accs.Read(ki.Stream(), binary, false);  // false == add; doesn't matter.
    }
    
    sgmm_num_accs.Check(am_sgmm, true); // Will check consistency and print some diagnostics.
    sgmm_den_accs.Check(am_sgmm, true); // Will check consistency and print some diagnostics.    

    {  // Update SGMM.
      BaseFloat auxf_impr, count;
      kaldi::EbwAmSgmm2Updater sgmm_updater(opts);
      sgmm_updater.Update(sgmm_num_accs, sgmm_den_accs, &am_sgmm,
                          update_flags, &auxf_impr, &count);
      KALDI_LOG << "Overall auxf impr/frame from SGMM update is " << (auxf_impr/count)
                << " over " << count << " frames.";
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_sgmm.Write(ko.Stream(), binary_write, write_flags);
    }
    
    KALDI_LOG << "Wrote model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

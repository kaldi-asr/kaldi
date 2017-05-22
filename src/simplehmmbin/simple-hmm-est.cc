// simplehmmbin/simple-hmm-est.cc

// Copyright 2009-2011  Microsoft Corporation
//                2016  Vimal Manohar

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
#include "simplehmm/simple-hmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Do Maximum Likelihood re-estimation of simple HMM "
        "transition parameters\n"
        "Usage:  simple-hmm-est [options] <model-in> <stats-in> <model-out>\n"
        "e.g.: simple-hmm-est 1.mdl 1.acc 2.mdl\n";

    bool binary_write = true;
    MleTransitionUpdateConfig tcfg;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    tcfg.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    SimpleHmm model;
    ReadKaldiObject(model_in_filename, &model);

    Vector<double> transition_accs;
    ReadKaldiObject(stats_filename, &transition_accs);

    {
      BaseFloat objf_impr, count;
      model.MleUpdate(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: Overall " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    WriteKaldiObject(model, model_out_filename, binary_write);

    if (GetVerboseLevel() >= 2) {
      std::vector<std::string> phone_names;
      phone_names.push_back("0");
      phone_names.push_back("1");
      model.Print(KALDI_LOG, phone_names);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



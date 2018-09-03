// sgmm2bin/sgmm2-sum-accs.cc

// Copyright 2009-2012   Saarland University;  Microsoft Corporation
//                       Johns Hopkins University (author: Daniel Povey)

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

#include "util/common-utils.h"
#include "sgmm2/estimate-am-sgmm2.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for SGMM training.\n"
        "Usage: sgmm2-sum-accs [options] stats-out stats-in1 stats-in2 ...\n";

    bool binary = true;
    bool parallel = false;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("parallel", &parallel, "If true, the program makes sure to open all "
                "filehandles before reading for any (useful when summing accs from "
                "long processes)");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_out_filename = po.GetArg(1);
    kaldi::Vector<double> transition_accs;
    kaldi::MleAmSgmm2Accs sgmm_accs;

    if (parallel) {
      std::vector<kaldi::Input*> inputs(po.NumArgs() - 1);
      for (int i = 0; i < po.NumArgs() - 1; i++) {
        std::string stats_in_filename = po.GetArg(i + 2);
        inputs[i] = new kaldi::Input(stats_in_filename); // Don't try
        // to work out binary status yet; this would cause us to wait
        // for the output of that process.  We delay it till later.
      }
      for (size_t i = 0; i < po.NumArgs() - 1; i++) {
        bool b;
        if (kaldi::InitKaldiInputStream(inputs[i]->Stream(), &b)) {
          transition_accs.Read(inputs[i]->Stream(), b, true /* add values */);
          sgmm_accs.Read(inputs[i]->Stream(), b, true /* add values */);
          delete inputs[i];
        } else {
          KALDI_ERR << "Failed to read input stats file " << po.GetArg(i + 2);
        }
      }      
    } else {
      for (int i = 2, max = po.NumArgs(); i <= max; i++) {
        std::string stats_in_filename = po.GetArg(i);
        bool binary_read;
        kaldi::Input ki(stats_in_filename, &binary_read);
        transition_accs.Read(ki.Stream(), binary_read, true /* add values */);
        sgmm_accs.Read(ki.Stream(), binary_read, true /* add values */);
      }
    }

    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
      transition_accs.Write(ko.Stream(), binary);
      sgmm_accs.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written stats to " << stats_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



// simplehmmbin/simple-hmm-acc-stats-ali.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
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
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate stats for simple HMM training.\n"
        "Usage:  simple-hmm-acc-stats-ali [options] <model-in> "
        "<alignments-rspecifier> <stats-out>\n"
        "e.g.:\n simple-hmm-acc-stats-ali 1.mdl ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2),
        accs_wxfilename = po.GetArg(3);

    SimpleHmm model;
    ReadKaldiObject(model_filename, &model);

    Vector<double> transition_accs;
    model.InitStats(&transition_accs);

    SequentialInt32VectorReader alignments_reader(alignments_rspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !alignments_reader.Done(); alignments_reader.Next()) {
      const std::string &key = alignments_reader.Key();
      const std::vector<int32> &alignment = alignments_reader.Value();

      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i];  // transition identifier.
        model.Accumulate(1.0, tid, &transition_accs);
      }
      
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



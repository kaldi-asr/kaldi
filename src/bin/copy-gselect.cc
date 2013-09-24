// bin/copy-gselect.cc

// Copyright 2009-2011   Saarland University;  Microsoft Corporation

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
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using std::vector;
    typedef kaldi::int32 int32;
    const char *usage =
        "Copy Gaussian indices for pruning, possibly making the\n"
        "lists shorter (e.g. the --n=10 limits to the 10 best indices\n"
        "Usage: \n"
        " copy-gselect [options] <gselect-rspecifier> <gselect-wspecifier>\n";
    
    ParseOptions po(usage);
    int32 num_gselect = 0;
    std::string likelihood_wspecifier;
    po.Register("n", &num_gselect, "Number of Gaussians to keep per frame (if nonzero)\n");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    KALDI_ASSERT(num_gselect >= 0);

    std::string gselect_rspecifier = po.GetArg(1),
        gselect_wspecifier = po.GetArg(2);

    SequentialInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    Int32VectorVectorWriter gselect_writer(gselect_wspecifier);
    int32 num_done = 0;
    for (; !gselect_reader.Done(); gselect_reader.Next()) {
      std::string utt = gselect_reader.Key();
      if (num_gselect == 0) { // keep original size.
        gselect_writer.Write(utt, gselect_reader.Value());
      } else {
        vector<vector<int32> > gselect(gselect_reader.Value());
        for (size_t i = 0; i < gselect.size(); i++)
          if (static_cast<int32>(gselect[i].size()) > num_gselect)
            gselect[i].resize(num_gselect); // keep 1st n elements.
        gselect_writer.Write(utt, gselect);
      }
      num_done++;
    }
    if (num_gselect == 0)
      KALDI_LOG << "Copied " << num_done << " gselect objects ";
    else
      KALDI_LOG << "Copied " << num_done << " gselect objects, "
                << " limiting sizes to " << num_gselect;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



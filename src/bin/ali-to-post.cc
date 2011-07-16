// bin/ali-to-post.cc

// Copyright 2009-2011  Microsoft Corporation, Go-Vivace Inc.

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"

/** @brief Convert alignments to viterbi style posteriors. The aligned
    symbol gets a weight of 1.0 */
int main(int argc, char *argv[])
{
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert alignments to posteriors\n"
        "Usage:  ali-to-post [options] alignments-rspecifier posteriors-wspecifier\n"
        "e.g.:\n"
        " ali-to-post ark:1.ali ark:1.post\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string alignments_rspecifier = po.GetArg(1);
    std::string posteriors_wspecifier = po.GetArg(2);

    int32 num_alignments = 0;
    SequentialInt32VectorReader alignment_reader(alignments_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      num_alignments++;
      const std::vector<int32> &alignment = alignment_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post(alignment.size());
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i];
        post[i].push_back(std::make_pair(tid, 1.0));
      }
      posterior_writer.Write(alignment_reader.Key(), post);
    }
    KALDI_LOG << "ali-to-post: converted " << num_alignments << " alignments.";
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}



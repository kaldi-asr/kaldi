// bin/get-post-on-ali.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "This program extracts a vector of per-frame posteriors that are selected\n"
        "by an alignment (ie. posteriors that are under the alignment path).\n"
        "This can be used as a per-frame confidence measure.\n"
        "\n"
        "By intuition, it is better to use pdf-posteriors and pdf-alignments,\n"
        "because the posteriors of competing hypothesis that are in the same frame\n"
        "at same 'pdf-state' are summed up, which is in some sense similar\n"
        "to what is done by C-max which sums the posteriors of overlapping words.\n"
        "The difference here is that the granularity is per-frame.\n"
        "\n"
        "Usage:  get-post-on-ali [options] <posteriors-rspecifier> <ali-rspecifier> <conf-wspecifier>\n"
        "e.g.: get-post-on-ali ark:post.ark ark:ali.ark ark:conf.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2),
        confidences_wspecifier = po.GetArg(3);

    int32 num_done = 0, num_no_alignment = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
    BaseFloatVectorWriter confidences_writer(confidences_wspecifier);
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();
      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        //get the posterior
        const kaldi::Posterior &posterior = posterior_reader.Value();
        int32 num_frames = static_cast<int32>(posterior.size());
        //get the alignment
        const std::vector<int32> &alignment = alignments_reader.Value(key);
        //check the lengths match
        KALDI_ASSERT(num_frames == alignment.size());

        //fill the vector with posteriors on the alignment (under the alignment path)
        Vector<BaseFloat> confidence(num_frames);
        for(int32 i = 0; i < num_frames; i++) {
          BaseFloat post_i = 0.0;
          for(int32 j = 0; j < posterior[i].size(); j++) {
            if(alignment[i] == posterior[i][j].first) {
              post_i = posterior[i][j].second;
              break;
            }
          }
          confidence(i) = post_i;
        }

        //write the vector with confidences
        confidences_writer.Write(key,confidence);
        num_done++;
      }
    }
    KALDI_LOG << "Done getting the posteriors under the alignment path for "
              << num_done << " utterances. " << num_no_alignment << " with missing alignments.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}





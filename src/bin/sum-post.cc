// bin/sum-post.cc

// Copyright 2011-2013 Johns Hopkins University (Author: Daniel Povey)  Chao Weng

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
#include "util/stl-utils.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Sum two sets of posteriors for each utterance, e.g. useful in fMMI.\n"
        "To take the difference of posteriors, use e.g. --scale2=-1.0\n"
        "\n"
        "Usage: sum-post <post-rspecifier1> <post-rspecifier2> <post-wspecifier>\n";

    BaseFloat scale1 = 1.0, scale2 = 1.0;
    bool merge = true;
    bool drop_frames = false;
    ParseOptions po(usage);
    po.Register("scale1", &scale1, "Scale for first set of posteriors");
    po.Register("scale2", &scale2, "Scale for second set of posteriors");
    po.Register("merge", &merge, "If true, merge posterior entries for "
                "same transition-id (canceling positive and negative parts)");
    po.Register("zero-if-disjoint", &drop_frames, "If true, zero "
                "posteriors on all frames when the two sets of posteriors are "
                "disjoint (this option for back-compatibility only; use "
                "'--drop-frames'");
    po.Register("drop-frames", &drop_frames, "If true, zero "
                "posteriors on all frames when the two sets of posteriors are "
                "disjoint");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string post_rspecifier1 = po.GetArg(1),
        post_rspecifier2 = po.GetArg(2),
        post_wspecifier = po.GetArg(3);

    kaldi::SequentialPosteriorReader posterior_reader1(post_rspecifier1);
    kaldi::RandomAccessPosteriorReader posterior_reader2(post_rspecifier2);
    kaldi::PosteriorWriter posterior_writer(post_wspecifier); 

    int32 num_done = 0, num_err = 0;
    int64 num_frames_tot = 0, num_frames_disjoint = 0;
   
    for (; !posterior_reader1.Done(); posterior_reader1.Next()) {
      std::string key = posterior_reader1.Key();
      kaldi::Posterior posterior1 = posterior_reader1.Value();
      if (!posterior_reader2.HasKey(key)) {
        KALDI_WARN << "Second set of posteriors has nothing for key "
                   << key << ", producing no output.";
        num_err++;
        continue;
      }
      kaldi::Posterior posterior2 = posterior_reader2.Value(key);
      if (posterior2.size() != posterior1.size()) {
        KALDI_WARN << "Posteriors have mismatched sizes " << posterior1.size()
                   << " vs. " << posterior2.size() << " for key " << key;
        num_err++;
        continue;
      }

      ScalePosterior(scale1, &posterior1);
      ScalePosterior(scale2, &posterior2);
      kaldi::Posterior posterior_out;
      num_frames_disjoint += MergePosteriors(posterior1, posterior2, merge,
                                             drop_frames, &posterior_out);
      num_frames_tot += static_cast<int64>(posterior1.size());
      posterior_writer.Write(key, posterior_out);
      num_done++;
    }
    KALDI_LOG << "Processed " << num_frames_tot << " frames; for "
              << num_frames_disjoint << " frames there was no overlap, i.e. "
              << (num_frames_disjoint * 100.0 / num_frames_tot)
              << "% (e.g. numerator path not in denominator lattice)";
    KALDI_LOG << "Done adding " << num_done << " posteriors;  " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


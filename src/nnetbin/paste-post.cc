// nnetbin/paste-post.cc

// Copyright 2015       Brno University of Technology (Author: Karel Vesely)

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
#include "base/io-funcs.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"
#include "nnet/nnet-utils.h"

/** @brief Combines 2 or more streams with NN-training targets into single one.
 *  This is handy when training NN with more than one output layer (softmax).
 *  The format of NN-targets is 'posterior' and the dimensionality of the output
 *  stream is the sum of input-stream dimensions.
 */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Combine 2 or more streams with NN-training targets into single stream.\n"
      "As the posterior streams are pasted, the output dimension is the sum\n"
      "of the input dimensions. This is used when training NN with\n"
      "multiple softmaxes on its output. This is used in multi-task, \n"
      "multi-lingual or multi-database training. Depending on the context,\n"
      "an utterance is not required to be in all the input streams.\n"
      "For a multi-database training only 1 output layer will be active.\n"
      "\n"
      "The lengths of utterances are provided as 1st argument.\n"
      "The dimensions of input stream are set as 2nd in argument.\n"
      "Follow the input and output streams which are in 'posterior' format.\n"
      "\n"
      "Usage: paste-post <featlen-rspecifier> <dims-csl> <post1-rspecifier> "
      "... <postN-rspecifier> <post-wspecifier>\n"
      "e.g.: paste-post 'ark:feat-to-len $feats ark,t:-|' 1029:1124 "
      "ark:post1.ark ark:post2.ark ark:pasted.ark\n";

    ParseOptions po(usage);

    bool allow_partial = false;
    po.Register("allow-partial", &allow_partial,
                "Produce output also when the utterance is not in all input streams.");

    po.Read(argc, argv);

    if (po.NumArgs() < 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string featlen_rspecifier = po.GetArg(1),  // segment lengths,
                stream_dims_str = po.GetArg(2),
                post_wspecifier = po.GetArg(po.NumArgs());
    int32 stream_count = po.NumArgs() - 3;  // number of input posterior streams

    // read the dims of input posterior streams,
    std::vector<int32> stream_dims;
    if (!kaldi::SplitStringToIntegers(stream_dims_str, ":,", false, &stream_dims)) {
      KALDI_ERR << "Invalid stream-dims string " << stream_dims_str;
    }
    if (stream_count != stream_dims.size()) {
      KALDI_ERR << "Mismatch in input posterior-stream count " << stream_count
                << " and --stream-dims count" << stream_dims.size()
                << ", " << stream_dims_str;
    }

    // prepare dim offsets of input streams,
    std::vector<int32> stream_offset(stream_dims.size()+1, 0);
    for (int32 s = 0; s < stream_dims.size(); s++) {
      stream_offset[s+1] = stream_offset[s] + stream_dims[s];
    }

    // open the input posterior readers,
    std::vector<RandomAccessPosteriorReader> posterior_reader(po.NumArgs()-3);
    for (int32 s = 0; s < stream_count; s++) {
      posterior_reader[s].Open(po.GetArg(s+3));
    }

    int32 num_done = 0, num_err = 0, num_empty = 0;
    SequentialInt32Reader featlen_reader(featlen_rspecifier);
    PosteriorWriter posterior_writer(post_wspecifier);

    // main loop, posterior pasting happens here,
    for (; !featlen_reader.Done(); featlen_reader.Next()) {
      bool ok = true, empty = true;
      std::string utt = featlen_reader.Key();
      int32 num_frames = featlen_reader.Value();

      // show which streams are non-empty,
      if (allow_partial && GetVerboseLevel() >= 2) {
        std::string nonempty_streams;
        for (int32 s = 0; s < stream_count; s++) {
          if (posterior_reader[s].HasKey(utt)) {
            nonempty_streams += " " + ToString(s);
          }
        }
        KALDI_VLOG(2) << "Processing " << utt
                      << ", frames " << num_frames
                      << ", pasted-from streams " << nonempty_streams;
      }

      // Create output posteriors,
      Posterior post(num_frames);

      // Fill posterior from input streams,
      for (int32 s = 0; s < stream_count; s++) {
        if (!posterior_reader[s].HasKey(utt)) {
          if (!allow_partial) {
            KALDI_WARN << "No such utterance " << utt
                       << " in set " << (s+1) << " of posteriors.";
            ok = false;
            break;
          }
        } else {
          const Posterior& post_s = posterior_reader[s].Value(utt);
          KALDI_ASSERT(num_frames <= post_s.size());
          for (int32 f = 0; f < num_frames; f++) {
            for (int32 i = 0; i < post_s[f].size(); i++) {
              int32 id = post_s[f][i].first;
              BaseFloat val = post_s[f][i].second;
              KALDI_ASSERT(id < stream_dims[s]);
              post[f].push_back(std::make_pair(stream_offset[s] + id, val));
            }
          }
          empty = false;
        }
      }
      if (empty) {
        KALDI_WARN << "Uttenrace with no posteriors " << utt << ", discarding";
        num_empty++;
        continue;
      }
      if (ok) {
        posterior_writer.Write(featlen_reader.Key(), post);
        num_done++;
      } else {
        num_err++;
      }
    }
    KALDI_LOG << "Pasted posteriors for " << num_done << " sentences, "
              << "missing sentences " << num_empty << ", "
              << "failed for " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}




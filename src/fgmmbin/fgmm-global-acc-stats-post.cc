// fgmmbin/fgmm-global-acc-stats-post.cc

// Copyright 2015 David Snyder
//           2015 Johns Hopkins University (Author: Daniel Povey)
//           2015 Johns Hopkins University (Author: Daniel Garcia-Romero)

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
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Accumulate stats from posteriors and features for instantiating "
        "a full-covariance GMM. See also fgmm-global-acc-stats.\n"
        "Usage:  fgmm-global-acc-stats-post [options] <posterior-rspecifier> "
        "<number-of-components> <feature-rspecifier> <stats-out>\n"
        "e.g.: fgmm-global-acc-stats-post scp:post.scp 2048 "
        "scp:train.scp 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string update_flags_str = "mvw";
    std::string weights_rspecifier;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters will be "
                "updated: subset of mvw.");
    po.Register("weights", &weights_rspecifier, "rspecifier for a vector of floats "
                "for each utterance, that's a per-frame weight.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string post_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    int32 num_components = atoi(po.GetArg(2).c_str());

    AccumFullGmm fgmm_accs;
    
    double tot_like = 0.0, tot_weight = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader post_reader(post_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    int32 num_done = 0, num_err = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      int32 file_frames = mat.NumRows();
      if (!post_reader.HasKey(key)) {
        KALDI_WARN << "No posteriors available for utterance "
                   << key;
        num_err++;
        continue;
      }

      Posterior post = post_reader.Value(key);     
      // Initialize the FGMM accs before processing the first utt.
      if (num_done == 0) {
        fgmm_accs.Resize(num_components, mat.NumCols(), 
          StringToGmmFlags(update_flags_str));
      }

      BaseFloat file_like = 0.0,
          file_weight = 0.0; // total of weights of frames (will each be 
                             // 1 unless --weights option supplied.
      Vector<BaseFloat> weights;
      if (weights_rspecifier != "") { // We have per-frame weighting.
        if (!weights_reader.HasKey(key)) {
          KALDI_WARN << "No per-frame weights available for utterance " 
                     << key;
          num_err++;
          continue;
        }
        weights = weights_reader.Value(key);
        if (weights.Dim() != file_frames) {
          KALDI_WARN << "Weights for utterance " << key << " have wrong dim "
                     << weights.Dim() << " vs. " << file_frames;
          num_err++;
          continue;
        }
      }

      if (post.size() != static_cast<size_t>(file_frames)) {
        KALDI_WARN << "posterior information for utterance " << key
                  << " has wrong size " << post.size() << " vs. "
                  << file_frames;
        num_err++;
        continue;
      }
        
      for (int32 i = 0; i < file_frames; i++) {
        BaseFloat weight = (weights.Dim() != 0) ? weights(i) : 1.0;
        if (weight == 0.0) continue;
        file_weight += weight;
        SubVector<BaseFloat> data(mat, i);
        ScalePosterior(weight, &post);
        file_like += TotalPosterior(post);
        for (int32 j = 0; j < post[i].size(); j++)
          fgmm_accs.AccumulateForComponent(data, post[i][j].first, 
            post[i][j].second);
      }

      KALDI_VLOG(2) << "File '" << key << "': Average likelihood = "
                    << (file_like/file_weight) << " over "
                    << file_weight <<" frames.";
      tot_like += file_like;
      tot_weight += file_weight;
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " files; "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per "
              << "frame = " << (tot_like/tot_weight) << " over " 
              << tot_weight << " (weighted) frames.";

    WriteKaldiObject(fgmm_accs, accs_wxfilename, binary);
    KALDI_LOG << "Written accs to " << accs_wxfilename;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

// gmmbin/gmm-global-get-post.cc

// Copyright 2009-2011   Saarland University;  Microsoft Corporation
//           2013-2014   Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using std::vector;
    typedef kaldi::int32 int32;
    const char *usage =
        "Precompute Gaussian indices and convert immediately to top-n\n"
        "posteriors (useful in iVector extraction with diagonal UBMs)\n"
        "See also: gmm-gselect, fgmm-gselect, fgmm-global-gselect-to-post\n"
        " (e.g. in training UBMs, SGMMs, tied-mixture systems)\n"
        " For each frame, gives a list of the n best Gaussian indices,\n"
        " sorted from best to worst.\n"
        "Usage: gmm-global-get-post [options] <model-in> <feature-rspecifier> <post-wspecifier>\n"
        "e.g.: gmm-global-get-post --n=20 1.gmm \"ark:feature-command |\" \"ark,t:|gzip -c >post.1.gz\"\n";
    
    ParseOptions po(usage);
    int32 num_post = 50;
    BaseFloat min_post = 0.0;
    po.Register("n", &num_post, "Number of Gaussians to keep per frame\n");
    po.Register("min-post", &min_post, "Minimum posterior we will output "
                "before pruning and renormalizing (e.g. 0.01)");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_wspecifier = po.GetArg(3);

    DiagGmm gmm;
    ReadKaldiObject(model_filename, &gmm);
    KALDI_ASSERT(num_post > 0);
    KALDI_ASSERT(min_post < 1.0);
    int32 num_gauss = gmm.NumGauss();
    if (num_post > num_gauss) {
      KALDI_WARN << "You asked for " << num_post << " Gaussians but GMM "
                 << "only has " << num_gauss << ", returning this many. ";
      num_post = num_gauss;
    }
    
    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    PosteriorWriter post_writer(post_wspecifier);
    
    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      int32 T = feats.NumRows();
      if (T == 0) {
        KALDI_WARN << "Empty features for utterance " << utt;
        num_err++;
        continue;
      }
      if (feats.NumCols() != gmm.Dim()) {
        KALDI_WARN << "Dimension mismatch for utterance " << utt
                   << ": got " << feats.NumCols() << ", expected " << gmm.Dim();
        num_err++;
        continue;
      }
      vector<vector<int32> > gselect(T);
      
      Matrix<BaseFloat> loglikes;
      
      gmm.LogLikelihoods(feats, &loglikes);

      Posterior post(T);

      double log_like_this_file = 0.0;
      for (int32 t = 0; t < T; t++) {
        log_like_this_file +=
            VectorToPosteriorEntry(loglikes.Row(t), num_post,
                                   min_post, &(post[t]));
      }
      KALDI_VLOG(1) << "Processed utterance " << utt << ", average likelihood "
                    << (log_like_this_file / T) << " over " << T << " frames";
      tot_like += log_like_this_file;
      tot_t += T;

      post_writer.Write(utt, post);
      num_done++;
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors, average UBM log-likelihood is "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";
    
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



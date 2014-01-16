// fgmmbin/fgmm-global-gselect-to-post.cc

// Copyright   2013       Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/full-gmm.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Given features and Gaussian-selection (gselect) information for\n"
        "a full-covariance GMM, output per-frame posteriors for the selected\n"
        "indices.  Also supports pruning the posteriors if they are below\n"
        "a stated threshold, (and renormalizing the rest to sum to one)\n"
        "\n"
        "Usage:  fgmm-global-gselect-to-post [options] <model-in> <feature-rspecifier> "
        "<gselect-rspecifier> <post-wspecifier>\n"
        "e.g.: fgmm-global-gselect-to-post 1.mdl ark:- 'ark:gunzip -c 1.gselect|' ark:-\n";
        
    ParseOptions po(usage);

    BaseFloat min_post = 0.0;
    po.Register("min-post", &min_post, "If nonzero, posteriors below this "
                "threshold will be pruned away and the rest will be renormalized "
                "to sum to one.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        gselect_rspecifier = po.GetArg(3),
        post_wspecifier = po.GetArg(4);
    
    FullGmm fgmm;
    ReadKaldiObject(model_rxfilename, &fgmm);
    
    double tot_loglike = 0.0, tot_frames = 0.0;
    int64 tot_posts = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    PosteriorWriter post_writer(post_wspecifier);
    int32 num_done = 0, num_err = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();

      int32 num_frames = mat.NumRows();
      // typedef std::vector<std::vector<std::pair<int32, BaseFloat> > > Posterior;
      Posterior post(num_frames);
      
      if (!gselect_reader.HasKey(utt)) {
        KALDI_WARN << "No gselect information for utterance " << utt;
        num_err++;
        continue;
      }
      const std::vector<std::vector<int32> > &gselect(gselect_reader.Value(utt));
      if (static_cast<int32>(gselect.size()) != num_frames) {
        KALDI_WARN << "gselect information for utterance " << utt
                   << " has wrong size " << gselect.size() << " vs. "
                   << num_frames;
        num_err++;
        continue;
      }

      double this_tot_loglike = 0;
      bool utt_ok = true;
      
      for (int32 t = 0; t < num_frames; t++) {
        SubVector<BaseFloat> frame(mat, t);
        const std::vector<int32> &this_gselect = gselect[t];
        KALDI_ASSERT(!gselect[t].empty());
        Vector<BaseFloat> loglikes;
        fgmm.LogLikelihoodsPreselect(frame, this_gselect, &loglikes);
        this_tot_loglike += loglikes.ApplySoftMax();
        // now "loglikes" contains posteriors.
        if (fabs(loglikes.Sum() - 1.0) > 0.01) {
          utt_ok = false;
        } else {
          if (min_post != 0.0) {
            int32 max_index = 0; // in case all pruned away...
            loglikes.Max(&max_index);
            for (int32 i = 0; i < loglikes.Dim(); i++)
              if (loglikes(i) < min_post)
                loglikes(i) = 0.0;
            BaseFloat sum = loglikes.Sum();
            if (sum == 0.0) {
              loglikes(max_index) = 1.0;
            } else {
              loglikes.Scale(1.0 / sum);
            }
          }
          for (int32 i = 0; i < loglikes.Dim(); i++) {
            if (loglikes(i) != 0.0) {
              post[t].push_back(std::make_pair(this_gselect[i], loglikes(i)));
              tot_posts++;
            }
          }
          KALDI_ASSERT(!post[t].empty());
        }
      }
      if (!utt_ok) {
        KALDI_WARN << "Skipping utterance " << utt
                  << " because bad posterior-sum encountered (NaN?)";
        num_err++;
      } else {
        post_writer.Write(utt, post);
        num_done++;
        KALDI_VLOG(2) << "Like/frame for utt " << utt << " was "
                      << (this_tot_loglike/num_frames) << " per frame over "
                      << num_frames << " frames.";
        tot_loglike += this_tot_loglike;
        tot_frames += num_frames;
      }
    }

    KALDI_LOG << "Done " << num_done << " files; " << num_err << " had errors.";
    KALDI_LOG << "Overall loglike per frame is " << (tot_loglike / tot_frames)
              << " with " << (tot_posts / tot_frames) << " entries per frame, "
              << " over " << tot_frames << " frames";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

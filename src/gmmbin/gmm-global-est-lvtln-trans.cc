// gmmbin/gmm-global-est-lvtln-trans.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/lvtln.h"
#include "hmm/posterior.h"

namespace kaldi {
void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const DiagGmm &gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());
  for (size_t i = 0; i < post.size(); i++) {
    std::vector<int32> gselect(post[i].size());
    Vector<BaseFloat> this_post(post[i].size());
    for (size_t j = 0; j < post[i].size(); j++) {
      int32 g = post[i][j].first;
      BaseFloat weight = post[i][j].second;
      gselect[j] = g;
      this_post(j) = weight;
    }
    spk_stats->AccumulateFromPosteriorsPreselect(gmm, gselect,
                                                 feats.Row(i),
                                                 this_post);
  }
}


}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate linear-VTLN transforms, either per utterance or for "
        "the supplied set of speakers (spk2utt option); this version\n"
        "is for a global diagonal GMM (also known as a UBM).  Reads posteriors\n"
        "indicating Gaussian indexes in the UBM.\n"
        "\n"
        "Usage: gmm-global-est-lvtln-trans [options] <gmm-in> <lvtln-in> "
        "<feature-rspecifier> <gpost-rspecifier> <lvtln-trans-wspecifier> [<warp-wspecifier>]\n"
        "e.g.: gmm-global-est-lvtln-trans 0.ubm 0.lvtln '$feats' ark,s,cs:- ark:1.trans ark:1.warp\n"
        "(where the <gpost-rspecifier> will likely come from gmm-global-get-post or\n"
        "gmm-global-gselect-to-post\n";
    
    ParseOptions po(usage);
    string spk2utt_rspecifier;
    BaseFloat logdet_scale = 1.0;
    std::string norm_type = "offset";
    po.Register("norm-type", &norm_type, "type of fMLLR applied (\"offset\"|\"none\"|\"diag\")");
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("logdet-scale", &logdet_scale, "Scale on log-determinant term in auxiliary function");

    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    string
        model_rxfilename = po.GetArg(1),
        lvtln_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        post_rspecifier = po.GetArg(4),
        trans_wspecifier = po.GetArg(5),
        warp_wspecifier = po.GetOptArg(6);

    DiagGmm gmm;
    ReadKaldiObject(model_rxfilename, &gmm);
    LinearVtln lvtln;
    ReadKaldiObject(lvtln_rxfilename, &lvtln);


    RandomAccessPosteriorReader post_reader(post_rspecifier);

    double tot_lvtln_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);

    BaseFloatWriter warp_writer(warp_wspecifier);

    std::vector<int32> class_counts(lvtln.NumClasses(), 0);
    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        FmllrDiagGmmAccs spk_stats(lvtln.Dim());
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            continue;
          }
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != feats.NumRows()) {
            KALDI_WARN << "Posterior vector has wrong size " << post.size()
                       << " vs. " << feats.NumRows();
            num_other_error++;
            continue;
          }

          AccumulateForUtterance(feats, post, gmm, &spk_stats);

          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(lvtln.Dim(), lvtln.Dim()+1);
          int32 class_idx;
          lvtln.ComputeTransform(spk_stats,
                                 norm_type,
                                 logdet_scale,
                                 &transform,
                                 &class_idx,
                                 NULL,
                                 &impr,
                                 &spk_tot_t);
          class_counts[class_idx]++;
          transform_writer.Write(spk, transform);
          if (warp_wspecifier != "")
            warp_writer.Write(spk, lvtln.GetWarp(class_idx));
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from LVTLN is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.";
        tot_lvtln_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posterior for utterance "
                     << utt;
          num_no_post++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const Posterior &post = post_reader.Value(utt);

        if (static_cast<int32>(post.size()) != feats.NumRows()) {
          KALDI_WARN << "Posterior has wrong size " << post.size()
              << " vs. " << feats.NumRows();
          num_other_error++;
          continue;
        }
        num_done++;

        FmllrDiagGmmAccs spk_stats(lvtln.Dim());

        AccumulateForUtterance(feats, post, gmm,
                               &spk_stats);
        BaseFloat impr, utt_tot_t = spk_stats.beta_;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(lvtln.Dim(), lvtln.Dim()+1);
          int32 class_idx;
          lvtln.ComputeTransform(spk_stats,
                                 norm_type,
                                 logdet_scale,
                                 &transform,
                                 &class_idx,
                                 NULL,
                                 &impr,
                                 &utt_tot_t);
          class_counts[class_idx]++;
          transform_writer.Write(utt, transform);
          if (warp_wspecifier != "")
            warp_writer.Write(utt, lvtln.GetWarp(class_idx));
        }

        KALDI_LOG << "For utterance " << utt << ", auxf-impr from LVTLN is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
        tot_lvtln_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    {
      std::ostringstream s;
      for (size_t i = 0; i < class_counts.size(); i++)
        s << ' ' << class_counts[i];
      KALDI_LOG << "Distribution of classes is: " << s.str();
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_post
              << " with no posteriors, " << num_other_error << " with other errors.";
    KALDI_LOG << "Overall LVTLN auxf impr per frame is "
              << (tot_lvtln_impr / tot_t) << " over " << tot_t << " frames.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


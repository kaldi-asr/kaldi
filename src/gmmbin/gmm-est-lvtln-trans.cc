// gmmbin/gmm-est-lvtln-trans.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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
                            const GaussPost &gpost,
                            const AmDiagGmm &am_gmm,
                            FmllrDiagGmmAccs *spk_stats) {
  for (size_t i = 0; i < gpost.size(); i++) {
    for (size_t j = 0; j < gpost[i].size(); j++) {
      int32 pdf_id = gpost[i][j].first;
      const Vector<BaseFloat> &posterior(gpost[i][j].second);
      spk_stats->AccumulateFromPosteriors(am_gmm.GetPdf(pdf_id),
                                          feats.Row(i), posterior);
    }
  }
}


}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate linear-VTLN transforms, either per utterance or for "
        "the supplied set of speakers (spk2utt option).  Reads posteriors. \n"
        "Usage: gmm-est-lvtln-trans [options] <model-in> <lvtln-in> "
        "<feature-rspecifier> <gpost-rspecifier> <lvtln-trans-wspecifier> [<warp-wspecifier>]\n";

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
        gpost_rspecifier = po.GetArg(4),
        trans_wspecifier = po.GetArg(5),
        warp_wspecifier = po.GetOptArg(6);

    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }
    LinearVtln lvtln;
    ReadKaldiObject(lvtln_rxfilename, &lvtln);


    RandomAccessGaussPostReader gpost_reader(gpost_rspecifier);

    double tot_lvtln_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);

    BaseFloatWriter warp_writer(warp_wspecifier);

    std::vector<int32> class_counts(lvtln.NumClasses(), 0);
    int32 num_done = 0, num_no_gpost = 0, num_other_error = 0;
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
          if (!gpost_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_gpost++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const GaussPost &gpost = gpost_reader.Value(utt);
          if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
            KALDI_WARN << "GauPost vector has wrong size " << (gpost.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          AccumulateForUtterance(feats, gpost, am_gmm, &spk_stats);

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
        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find gposts for utterance "
                     << utt;
          num_no_gpost++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const GaussPost &gpost = gpost_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "GauPost has wrong size " << (gpost.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        num_done++;

        FmllrDiagGmmAccs spk_stats(lvtln.Dim());

        AccumulateForUtterance(feats, gpost, am_gmm,
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

    KALDI_LOG << "Done " << num_done << " files, " << num_no_gpost
              << " with no gposts, " << num_other_error << " with other errors.";
    KALDI_LOG << "Overall LVTLN auxf impr per frame is "
              << (tot_lvtln_impr / tot_t) << " over " << tot_t << " frames.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


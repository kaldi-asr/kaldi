// gmmbin/gmm-est-fmllr-raw-gpost.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)
//           2014  Guoguo Chen

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
#include "transform/fmllr-raw.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"

namespace kaldi {


void AccStatsForUtterance(const TransitionModel &trans_model,
                          const AmDiagGmm &am_gmm,
                          const GaussPost &gpost,
                          const Matrix<BaseFloat> &feats,
                          FmllrRawAccs *accs) {
  for (size_t t = 0; t < gpost.size(); t++) {
    for (size_t i = 0; i < gpost[t].size(); i++) {
      int32 pdf = gpost[t][i].first;
      const Vector<BaseFloat> &posterior(gpost[t][i].second);      
      accs->AccumulateFromPosteriors(am_gmm.GetPdf(pdf),
                                     feats.Row(t), posterior);
    }
  }
}


}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate fMLLR transforms in the space before splicing and linear transforms\n"
        "such as LDA+MLLT, but using models in the space transformed by these transforms\n"
        "Requires the original spliced features, and the full LDA+MLLT (or similar) matrix\n"
        "including the 'rejected' rows (see the program get-full-lda-mat).  Reads in\n"
        "Gaussian-level posteriors.\n"
        "Usage: gmm-est-fmllr-raw-gpost [options] <model-in> <full-lda-mat-in> "
        "<feature-rspecifier> <gpost-rspecifier> <transform-wspecifier>\n";


    int32 raw_feat_dim = 13;
    ParseOptions po(usage);
    FmllrRawOptions opts;
    std::string spk2utt_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("raw-feat-dim", &raw_feat_dim, "Dimension of raw features "
                "prior to splicing");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string model_rxfilename = po.GetArg(1),
        full_lda_mat_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        gpost_rspecifier = po.GetArg(4),
        transform_wspecifier = po.GetArg(5);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Matrix<BaseFloat> full_lda_mat;
    ReadKaldiObject(full_lda_mat_rxfilename, &full_lda_mat);
    
    RandomAccessGaussPostReader gpost_reader(gpost_rspecifier);
    BaseFloatMatrixWriter transform_writer(transform_wspecifier);
    
    double tot_auxf_impr = 0.0, tot_count = 0.0;
    
    int32 num_done = 0, num_err = 0;
    if (!spk2utt_rspecifier.empty()) { // Adapting per speaker
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        FmllrRawAccs accs(raw_feat_dim, am_gmm.Dim(), full_lda_mat);
        std::string spk = spk2utt_reader.Key();
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Features not found for utterance " << utt;
            num_err++;
            continue;
          }
          if (!gpost_reader.HasKey(utt)) {
            KALDI_WARN << "Gaussian-level posteriors not found for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const GaussPost &gpost = gpost_reader.Value(utt);
          if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
            KALDI_WARN << "Size mismatch between gposteriors " << gpost.size()
                       << " and features " << feats.NumRows();
            num_err++;
            continue;
          }

          AccStatsForUtterance(trans_model, am_gmm, gpost, feats, &accs);
          num_done++;
        }
        
        BaseFloat auxf_impr, count;
        {
          Matrix<BaseFloat> transform(raw_feat_dim, raw_feat_dim + 1);
          transform.SetUnit();
          accs.Update(opts, &transform, &auxf_impr, &count);
          transform_writer.Write(spk, transform);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from raw fMLLR is "
                  << (auxf_impr/count) << " over " << count << " frames.";
        tot_auxf_impr += auxf_impr;
        tot_count += count;
      }
    } else {  // --spk2utt option not given -> adapt per utterance.
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        if (!gpost_reader.HasKey(utt)) {
          KALDI_WARN << "Gaussian-level posteriors not found for utterance " << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const GaussPost &gpost = gpost_reader.Value(utt);

        if (static_cast<int32>(gpost.size()) != feats.NumRows()) {
          KALDI_WARN << "Size mismatch between posteriors " << gpost.size()
                     << " and features " << feats.NumRows();
          num_err++;
          continue;
        }

        FmllrRawAccs accs(raw_feat_dim, am_gmm.Dim(), full_lda_mat);

        AccStatsForUtterance(trans_model, am_gmm, gpost, feats, &accs);
        
        BaseFloat auxf_impr, count;        
        {
          Matrix<BaseFloat> transform(raw_feat_dim, raw_feat_dim + 1);
          transform.SetUnit();
          accs.Update(opts, &transform, &auxf_impr, &count);
          transform_writer.Write(utt, transform);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from raw fMLLR is "
                  << (auxf_impr/count) << " over " << count << " frames.";
        tot_auxf_impr += auxf_impr;
        tot_count += count;
        num_done++;
      }
    }

    KALDI_LOG << "Processed " << num_done << " utterances, "
              << num_err << " had errors.";
    KALDI_LOG << "Overall raw-fMLLR auxf impr per frame is "
              << (tot_auxf_impr / tot_count) << " over " << tot_count
              << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


// gmmbin/gmm-est-regtree-fmllr.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation
//                2014  Guoguo Chen

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
#include "hmm/posterior.h"
#include "transform/regtree-fmllr-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Compute FMLLR transforms per-utterance (default) or per-speaker for "
        "the supplied set of speakers (spk2utt option).  Note: writes RegtreeFmllrDiagGmm objects\n"
        "Usage: gmm-est-regtree-fmllr  [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <regression-tree> <transforms-wspecifier>\n";

    ParseOptions po(usage);
    string spk2utt_rspecifier;
    bool binary = true;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("binary", &binary, "Write output in binary mode");
    // register other modules
    RegtreeFmllrOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        regtree_filename = po.GetArg(4),
        xforms_wspecifier = po.GetArg(5);

    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    RegtreeFmllrDiagGmmWriter fmllr_writer(xforms_wspecifier);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }
    RegressionTree regtree;
    {
      bool binary;
      Input in(regtree_filename, &binary);
      regtree.Read(in.Stream(), binary, am_gmm);
    }

    RegtreeFmllrDiagGmm fmllr_xforms;
    RegtreeFmllrDiagGmmAccs fmllr_accs;
    fmllr_accs.Init(regtree.NumBaseclasses(), am_gmm.Dim());

    double tot_like = 0.0, tot_t = 0;

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    double tot_objf_impr = 0.0, tot_t_objf = 0.0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        string spk = spk2utt_reader.Key();
        fmllr_accs.SetZero();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (vector<string>::const_iterator utt_itr = uttlist.begin(),
            itr_end = uttlist.end(); utt_itr != itr_end; ++utt_itr) {
          if (!feature_reader.HasKey(*utt_itr)) {
            KALDI_WARN << "Did not find features for utterance " << *utt_itr;
            continue;
          }
          if (!posteriors_reader.HasKey(*utt_itr)) {
            KALDI_WARN << "Did not find posteriors for utterance "
                << *utt_itr;
            num_no_posterior++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(*utt_itr);
          const Posterior &posterior = posteriors_reader.Value(*utt_itr);
          if (static_cast<int32>(posterior.size()) != feats.NumRows()) {
            KALDI_WARN << "Posteriors has wrong size " << (posterior.size())
                << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          BaseFloat file_like = 0.0, file_t = 0.0;
          Posterior pdf_posterior;
          ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
          for (size_t i = 0; i < posterior.size(); i++) {
            for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
              int32 pdf_id = pdf_posterior[i][j].first;
              BaseFloat prob = pdf_posterior[i][j].second;
              file_like += fmllr_accs.AccumulateForGmm(regtree, am_gmm,
                                                       feats.Row(i), pdf_id,
                                                       prob);
              file_t += prob;
            }
          }
          KALDI_VLOG(2) << "Average like for this file is " << (file_like/file_t)
                        << " over " << file_t << " frames.";
          tot_like += file_like;
          tot_t += file_t;
          num_done++;
          if (num_done % 10 == 0)
            KALDI_VLOG(1) << "Avg like per frame so far is "
                          << (tot_like / tot_t);
        }  // end looping over all utterances of the current speaker
        BaseFloat objf_impr, t;
        fmllr_accs.Update(regtree, opts, &fmllr_xforms, &objf_impr, &t);
        KALDI_LOG << "fMLLR objf improvement for speaker " << spk << " is "
                  << (objf_impr/(t+1.0e-10)) << " per frame over " << t
                  << " frames.";
        tot_objf_impr += objf_impr;
        tot_t_objf += t;
        fmllr_writer.Write(spk, fmllr_xforms);
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string key = feature_reader.Key();
        if (!posteriors_reader.HasKey(key)) {
          KALDI_WARN << "Did not find posteriors for utterance "
              << key;
          num_no_posterior++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        if (static_cast<int32>(posterior.size()) != feats.NumRows()) {
          KALDI_WARN << "Posteriors has wrong size " << (posterior.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat file_like = 0.0, file_t = 0.0;
        fmllr_accs.SetZero();
        Posterior pdf_posterior;
        ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
        for (size_t i = 0; i < posterior.size(); i++) {
          for (size_t j = 0; j < pdf_posterior[i].size(); j++) {
            int32 pdf_id = pdf_posterior[i][j].first;
            BaseFloat prob = pdf_posterior[i][j].second;
            file_like += fmllr_accs.AccumulateForGmm(regtree, am_gmm,
                                                     feats.Row(i), pdf_id,
                                                     prob);
            file_t += prob;
          }
        }
        KALDI_VLOG(2) << "Average like for this file is " << (file_like/file_t)
                      << " over " << file_t << " frames.";
        tot_like += file_like;
        tot_t += file_t;
        if (num_done % 10 == 0)
          KALDI_VLOG(1) << "Avg like per frame so far is "
                        << (tot_like / tot_t);
        BaseFloat objf_impr, t;
        fmllr_accs.Update(regtree, opts, &fmllr_xforms, &objf_impr, &t);
        KALDI_LOG << "fMLLR objf improvement for utterance " << key << " is "
                  << (objf_impr/(t+1.0e-10)) << " per frame over " << t
                  << " frames.";
        tot_objf_impr += objf_impr;
        tot_t_objf += t;
        fmllr_writer.Write(feature_reader.Key(), fmllr_xforms);
      }
    }
    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";
    KALDI_LOG << "Overall objf improvement from MLLR is " << (tot_objf_impr/tot_t_objf)
              << " per frame " << " over " << tot_t_objf << " frames.";
    KALDI_LOG << "Overall acoustic likelihood was " << (tot_like/tot_t)
              << " over " << tot_t << " frames.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


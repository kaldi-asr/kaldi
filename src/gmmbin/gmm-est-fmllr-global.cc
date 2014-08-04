// gmmbin/gmm-est-fmllr.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
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
#include "transform/fmllr-diag-gmm.h"
#include "hmm/posterior.h"



int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate global fMLLR transforms, either per utterance or for the supplied\n"
        "set of speakers (spk2utt option).  This version is for when you have a single\n"
        "global GMM, e.g. a UBM.  Writes to a table of matrices.\n"
        "Usage: gmm-est-fmllr-global [options] <gmm-in> <feature-rspecifier> "
        "<transform-wspecifier>\n"
        "e.g.: gmm-est-fmllr-global 1.ubm scp:feats.scp ark:trans.1\n";
    
    ParseOptions po(usage);
    FmllrOptions fmllr_opts;
    string spk2utt_rspecifier;
    string gselect_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("gselect", &gselect_rspecifier, "rspecifier for "
                "Gaussian-selection information");
    fmllr_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string gmm_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        trans_wspecifier = po.GetArg(3);

    DiagGmm gmm;
    ReadKaldiObject(gmm_rxfilename, &gmm);

    double tot_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    int32 num_done = 0, num_err = 0;
    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        FmllrDiagGmmAccs spk_stats(gmm.Dim(), fmllr_opts);
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);

          if (gselect_rspecifier == "") {
            for (size_t i = 0; i < feats.NumRows(); i++)
              spk_stats.AccumulateForGmm(gmm, feats.Row(i), 1.0);
          } else {
            if (!gselect_reader.HasKey(utt) ||
                gselect_reader.Value(utt).size() != feats.NumRows()) {
              KALDI_LOG << "No gselect information for utterance " << utt
                        << " (or wrong size)";
              num_err++;
              continue;
            }
            const std::vector<std::vector<int32> > &gselect =
                gselect_reader.Value(utt);
            for (size_t i = 0; i < feats.NumRows(); i++)
              spk_stats.AccumulateForGmmPreselect(gmm, gselect[i],
                                                  feats.Row(i), 1.0);
          }
          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(gmm.Dim(), gmm.Dim()+1);
          transform.SetUnit();
          spk_stats.Update(fmllr_opts, &transform, &impr, &spk_tot_t);
          transform_writer.Write(spk, transform);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from fMLLR is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        const Matrix<BaseFloat> &feats = feature_reader.Value();

        num_done++;

        FmllrDiagGmmAccs spk_stats(gmm.Dim(), fmllr_opts);

        if (gselect_rspecifier == "") {
          for (size_t i = 0; i < feats.NumRows(); i++)
            spk_stats.AccumulateForGmm(gmm, feats.Row(i), 1.0);
        } else {
          if (!gselect_reader.HasKey(utt) ||
              gselect_reader.Value(utt).size() != feats.NumRows()) {
            KALDI_LOG << "No gselect information for utterance " << utt
                      << " (or wrong size)";
            num_err++;
            continue;
          }
          const std::vector<std::vector<int32> > &gselect =
              gselect_reader.Value(utt);
          for (size_t i = 0; i < feats.NumRows(); i++)
            spk_stats.AccumulateForGmmPreselect(gmm, gselect[i],
                                                feats.Row(i), 1.0);
        }
        BaseFloat impr, utt_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(gmm.Dim(), gmm.Dim()+1);
          transform.SetUnit();
          spk_stats.Update(fmllr_opts, &transform, &impr, &utt_tot_t);
          transform_writer.Write(utt, transform);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from fMLLR is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }
    KALDI_LOG << "Done " << num_done << " files, " 
              << num_err << " with errors.";
    KALDI_LOG << "Overall fMLLR auxf impr per frame is "
              << (tot_impr / tot_t) << " over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


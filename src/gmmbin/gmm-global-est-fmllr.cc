// gmmbin/gmm-global-est-fmllr.cc

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
#include "transform/fmllr-diag-gmm.h"

namespace kaldi {
bool AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const DiagGmm &gmm,
                            const std::string &key,
                            RandomAccessBaseFloatVectorReader *weights_reader,
                            RandomAccessInt32VectorVectorReader *gselect_reader,
                            AccumFullGmm *fullcov_stats) {
  Vector<BaseFloat> weights;
  if (weights_reader->IsOpen()) {
    if (!weights_reader->HasKey(key)) {
      KALDI_WARN << "No weights present for utterance " << key;
      return false;
    }
    weights = weights_reader->Value(key);
  }
  int32 num_frames = feats.NumRows();
  if (gselect_reader->IsOpen()) {
    if (!gselect_reader->HasKey(key)) {
      KALDI_WARN << "No gselect information present for utterance " << key;
      return false;
    }
    const std::vector<std::vector<int32> > &gselect(gselect_reader->Value(key));
    if (gselect.size() != num_frames) {
      KALDI_WARN << "gselect information has wrong size for utterance " << key;
      return false;
    }
    for (int32 t = 0; t < num_frames; t++) {
      const std::vector<int32> &this_gselect(gselect[t]);
      BaseFloat weight = (weights.Dim() != 0 ? weights(t) : 1.0);
      if (weight != 0.0) {
        Vector<BaseFloat> post(this_gselect.size());
        gmm.LogLikelihoodsPreselect(feats.Row(t), this_gselect, &post);
        post.ApplySoftMax(); // get posteriors.
        post.Scale(weight); // scale by the weight for this frame.
        for (size_t i = 0; i < this_gselect.size(); i++)
          fullcov_stats->AccumulateForComponent(feats.Row(t),
                                                this_gselect[i], post(i));
      }
    }
  } else {
    for (int32 t = 0; t < num_frames; t++) {
      BaseFloat weight = (weights.Dim() != 0 ? weights(t) : 1.0);
      if (weight != 0.0)
        fullcov_stats->AccumulateFromDiag(gmm, feats.Row(t), weight);
    }
  }
  return true;
}
      

}

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate global fMLLR transforms, either per utterance or for the supplied\n"
        "set of speakers (spk2utt option).  Reads features, and (with --weights option)\n"
        "weights for each frame (also see --gselect option)\n"
        "Usage: gmm-global-est-fmllr [options] <gmm-in> <feature-rspecifier> <transform-wspecifier>\n";

    ParseOptions po(usage);
    FmllrOptions fmllr_opts;
    string spk2utt_rspecifier, gselect_rspecifier, weights_rspecifier,
        alignment_model;
        

    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    po.Register("gselect", &gselect_rspecifier, "rspecifier for gselect objects "
                "to limit the #Gaussians accessed on each frame.");
    po.Register("weights", &weights_rspecifier, "rspecifier for a vector of floats "
                "for each utterance, that's a per-frame weight.");
    po.Register("align-model", &alignment_model, "rxfilename for a model in the "
                "speaker-independent space, to get Gaussian alignments from");
    
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
    DiagGmm ali_gmm_read;
    if (alignment_model != "") {
      bool binary;
      Input ki(gmm_rxfilename, &binary);
      ali_gmm_read.Read(ki.Stream(), binary);
    }
    DiagGmm &ali_gmm = (alignment_model != "" ? ali_gmm_read : gmm);
    
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    double tot_impr = 0.0, tot_t = 0.0;

    BaseFloatMatrixWriter transform_writer(trans_wspecifier);

    int32 num_done = 0, num_err = 0;

    if (spk2utt_rspecifier != "") {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        AccumFullGmm fullcov_stats(gmm.NumGauss(), gmm.Dim(), kGmmAll);
        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);

          if (AccumulateForUtterance(feats, ali_gmm, utt, &weights_reader,
                                     &gselect_reader, &fullcov_stats)) num_done++;
          else num_err++;
        }  // end looping over all utterances of the current speaker
        
        BaseFloat impr, spk_tot_t;
        {  // Compute the transform and write it out.
          Matrix<BaseFloat> transform(gmm.Dim(), gmm.Dim()+1);
          transform.SetUnit();
          FmllrDiagGmmAccs spk_stats(gmm, fullcov_stats);
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

        AccumFullGmm fullcov_stats(gmm.NumGauss(), gmm.Dim(), kGmmAll);

        if (AccumulateForUtterance(feats, ali_gmm, utt, &weights_reader,
                                   &gselect_reader, &fullcov_stats)) {
          BaseFloat impr, utt_tot_t;
          {  // Compute the transform and write it out.
            Matrix<BaseFloat> transform(gmm.Dim(), gmm.Dim()+1);
            transform.SetUnit();
            FmllrDiagGmmAccs spk_stats(gmm, fullcov_stats);
            spk_stats.Update(fmllr_opts, &transform, &impr, &utt_tot_t);
            transform_writer.Write(utt, transform);
          }
          KALDI_LOG << "For utterance " << utt << ", auxf-impr from fMLLR is "
                    << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
          tot_impr += impr;
          tot_t += utt_tot_t;
          num_done++;
        } else num_err++;
        
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall fMLLR auxf impr per frame is "
              << (tot_impr / tot_t) << " over " << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


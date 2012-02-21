// gmmbin/gmm-fmpe-acc-stats.cc

// Copyright 2012  Daniel Povey

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "transform/fmpe.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using kaldi::int32;
  try {
    const char *usage =
        "Accumulate stats for fMPE training, using GMM model.  Note: this could\n"
        "be done using gmm-get-feat-deriv and fmpe-acc-stats (but you'd be computing\n"
        "the features twice).  Features input should be pre-fMPE features.\n"
        "\n"
        "Usage:  gmm-fmpe-acc-stats [options] <model-in> <fmpe-in> <feature-rspecifier> "
        "<gselect-rspecifier> <posteriors-rspecifier> <fmpe-stats-out>\n"
        "e.g.: \n"
        " gmm-fmpe-acc-stats 1.mdl 1.fmpe \"$feats\" ark:1.gselect ark:1.post 1.fmpe_stats\n";
        
    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "If true, write stats in binary mode.");
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        fmpe_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        gselect_rspecifier = po.GetArg(4),
        posteriors_rspecifier = po.GetArg(5),
        stats_wxfilename = po.GetArg(6);
    
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Fmpe fmpe;
    {
      bool binary_in;
      Input ki(fmpe_rxfilename, &binary_in);
      fmpe.Read(ki.Stream(), binary_in);
    }

    // fmpe stats...
    Matrix<BaseFloat> stats(fmpe.ProjectionTNumRows() * 2,
                            fmpe.ProjectionTNumCols());
    SubMatrix<BaseFloat> stats_plus(stats, 0, fmpe.ProjectionTNumRows(),
                                    0, fmpe.ProjectionTNumCols());
    SubMatrix<BaseFloat> stats_minus(stats, fmpe.ProjectionTNumRows(),
                                    fmpe.ProjectionTNumRows(),
                                    0, fmpe.ProjectionTNumCols());

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    BaseFloat tot_like = 0.0; // tot like weighted by posterior.
    int32 num_frames = 0;
    int32 num_done = 0, num_err = 0;
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!posteriors_reader.HasKey(key)) {
        num_err++;
        KALDI_WARN << "No posteriors for utterance " << key;
        continue;
      } 
      const Matrix<BaseFloat> &feat_in = feature_reader.Value();
      const Posterior &posterior = posteriors_reader.Value(key);

      if (static_cast<int32>(posterior.size()) != feat_in.NumRows()) {
        KALDI_WARN << "Posterior vector has wrong size " <<
            (posterior.size()) << " vs. "<< (feat_in.NumRows());
        num_err++;
        continue;
      }

      if (!gselect_reader.HasKey(key)) {
        KALDI_WARN << "No gselect information for key " << key;
        num_err++;
        continue;
      }
      const std::vector<std::vector<int32> > &gselect =
          gselect_reader.Value(key);
      if (static_cast<int32>(gselect.size()) != feat_in.NumRows()) {
        KALDI_WARN << "gselect information has wrong size";
        num_err++;
        continue;
      }
      
      num_done++;
      Matrix<BaseFloat> fmpe_feat(feat_in.NumRows(), feat_in.NumCols());
      fmpe.ComputeFeatures(feat_in, gselect, &fmpe_feat);
      fmpe_feat.AddMat(1.0, feat_in);
      
      Matrix<BaseFloat> feat_deriv;

      tot_like += ComputeAmGmmFeatureDeriv(am_gmm, trans_model, posterior,
                                           fmpe_feat, &feat_deriv);
      num_frames += feat_in.NumRows();

      fmpe.AccStats(feat_in, gselect, feat_deriv, &stats_plus, &stats_minus);
      
      if (num_done % 100 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances.";
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall weighted acoustic likelihood per frame is "
              << (tot_like/num_frames) << " over " << num_frames << " frames.";

    Output ko(stats_wxfilename, binary);
    stats.Write(ko.Stream(), binary);
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}



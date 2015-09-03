// gmmbin/gmm-fmpe-acc-stats.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
        " gmm-fmpe-acc-stats --model-derivative 1.accs 1.mdl 1.fmpe \"$feats\" ark:1.gselect ark:1.post 1.fmpe_stats\n";
        
    ParseOptions po(usage);
    bool binary = true;
    std::string model_derivative_rxfilename;
    po.Register("binary", &binary, "If true, write stats in binary mode.");
    po.Register("model-derivative", &model_derivative_rxfilename,
                "GMM-accs file containing model derivative [note: contains no transition stats].  Used for indirect differential.  Warning: this will only work correctly in the case of MMI/BMMI objective function, with non-canceled stats.");
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
    ReadKaldiObject(fmpe_rxfilename, &fmpe);


    bool have_indirect = (model_derivative_rxfilename != "");
    AccumAmDiagGmm model_derivative;
    if (have_indirect)
      ReadKaldiObject(model_derivative_rxfilename, &model_derivative);
    
    FmpeStats fmpe_stats(fmpe);
    
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
      
      Matrix<BaseFloat> direct_deriv, indirect_deriv;

      tot_like += ComputeAmGmmFeatureDeriv(am_gmm, trans_model, posterior,
                                           fmpe_feat, &direct_deriv,
                                           (have_indirect ? &model_derivative : NULL),
                                           (have_indirect ? &indirect_deriv : NULL));
      num_frames += feat_in.NumRows();

      fmpe.AccStats(feat_in, gselect, direct_deriv,
                    (have_indirect ? &indirect_deriv : NULL), &fmpe_stats);
      
      if (num_done % 100 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances.";
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Overall weighted acoustic likelihood per frame is "
              << (tot_like/num_frames) << " over " << num_frames << " frames.";

    Output ko(stats_wxfilename, binary);
    fmpe_stats.Write(ko.Stream(), binary);
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



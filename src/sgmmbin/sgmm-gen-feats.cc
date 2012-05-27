// sgmmbin/sgmm-gen-feats.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Generate features based on derivative w.r.t. SGMM parameters (an experimental feature).\n"
        "Usage:  sgmm-gen-feats [options] <model-in> <features-rspecifier> <features-wspecifier>\n";
    ParseOptions po(usage);

    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;

    po.Register("gselect", &gselect_rspecifier,
                "rspecifier for precomputed per-frame Gaussian indices [required]");
    po.Register("spk-vecs", &spkvecs_rspecifier,
                "rspecifier for speaker vectors");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        features_rspecifier = po.GetArg(2),
        features_wspecifier = po.GetArg(3);
    
    TransitionModel trans_model;
    kaldi::AmSgmm am_sgmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is mandatory.";
    
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);
    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);

    double tot_like = 0.0;
    int64 frame_count = 0;
    int32 num_done = 0, num_err = 0;

    SgmmFeature feature_computer(am_sgmm);
    
    SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
    BaseFloatMatrixWriter feature_writer(features_wspecifier);
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &input_features = feature_reader.Value();
      if (!gselect_reader.HasKey(utt) ||
          gselect_reader.Value(utt).size() != input_features.NumRows()) {
        KALDI_WARN << "No gselect info available for utterance " << utt
                   << " (or wrong size)";
        num_err++;
        continue;
      }

      const std::vector<std::vector<int32> > &gselect = gselect_reader.Value(utt);

      std::string utt_or_spk;  
      if (utt2spk_reader.IsOpen()) {
        if (!utt2spk_reader.HasKey(utt)) {
          KALDI_WARN << "Utterance " << utt << " not present in utt2spk map; "
                     << "skipping this utterance.";
          num_err++;
          continue;
        } else { utt_or_spk = utt2spk_reader.Value(utt); }
      } else { utt_or_spk = utt; }
      
      SgmmPerSpkDerivedVars spk_vars;
      if (spkvecs_reader.IsOpen()) {
        if (spkvecs_reader.HasKey(utt_or_spk)) {
          spk_vars.v_s = spkvecs_reader.Value(utt_or_spk);
          am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
        } else {
          KALDI_WARN << "Cannot find speaker vector for "
                     << utt_or_spk;
          num_err++;
          continue;
        }
      }

      SgmmPerFrameDerivedVars per_frame_vars;
      int32 T = input_features.NumRows();
      Matrix<BaseFloat> sgmm_feats(T, am_sgmm.PhoneSpaceDim());
      for (int32 t = 0; t < T; t++) {
        am_sgmm.ComputePerFrameVars(input_features.Row(t),
                                    gselect[t], spk_vars, 0.0, &per_frame_vars);
        SubVector<BaseFloat> feature_row(sgmm_feats, t);
        tot_like += feature_computer.ComputeFeature(per_frame_vars,
                                                    &feature_row);
      }
      frame_count += T;
      feature_writer.Write(utt, sgmm_feats);
      num_done++;
      if (num_done % 100 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances.";
    }
    KALDI_LOG << "Done " << num_done << " utterances, errors on " << num_err;
    KALDI_LOG << "Average log-like per frame given single speaker-vector was "
              << (tot_like/frame_count) << " over " << frame_count << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}



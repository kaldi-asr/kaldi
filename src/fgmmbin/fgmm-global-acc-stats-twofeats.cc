// fgmmbin/fgmm-global-acc-stats-twofeats.cc

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Accumulate stats for training a full-covariance GMM, two-feature version\n"
        "Usage:  fgmm-global-acc-stats-twofeats [options] <model-in> "
        "<feature1-rspecifier> <feature2-rspecifier> <stats-out>\n"
        "e.g.: fgmm-global-acc-stats-twofeats 1.mdl scp:train.scp scp:train2.scp 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string update_flags_str = "mvw";
    std::string gselect_rspecifier, weights_rspecifier;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters will be "
                "updated: subset of mvw.");
    po.Register("gselect", &gselect_rspecifier, "rspecifier for gselect objects "
                "to limit the #Gaussians accessed on each frame.");
    po.Register("weights", &weights_rspecifier, "rspecifier for a vector of floats "
                "for each utterance, that's a per-frame weight.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature1_rspecifier = po.GetArg(2),
        feature2_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    FullGmm gmm;
    {
      bool binary_read;
      Input ki(model_filename, &binary_read);
      gmm.Read(ki.Stream(), binary_read);
    }

    int32 new_dim = 0;
    AccumFullGmm gmm_accs;
    // will initialize once we know new_dim.    
    // gmm_accs.Resize(gmm, StringToGmmFlags(update_flags_str));
    
    double tot_like = 0.0, tot_weight = 0.0;

    SequentialBaseFloatMatrixReader feature1_reader(feature1_rspecifier);
    RandomAccessBaseFloatMatrixReader feature2_reader(feature2_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    int32 num_done = 0, num_err = 0;

    for (; !feature1_reader.Done(); feature1_reader.Next()) {
      std::string key = feature1_reader.Key();
      if (!feature2_reader.HasKey(key)) {
        KALDI_WARN << "For utterance " << key << ", second features not present.";
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &mat1 = feature1_reader.Value();
      const Matrix<BaseFloat> &mat2 = feature2_reader.Value(key);
      int32 file_frames = mat1.NumRows();
      KALDI_ASSERT(mat1.NumRows() == mat2.NumRows());
      if (new_dim == 0) {
        new_dim = mat2.NumCols();
        gmm_accs.Resize(gmm.NumGauss(), new_dim,
                        StringToGmmFlags(update_flags_str));
      }
      BaseFloat file_like = 0.0,
          file_weight = 0.0; // total of weights of frames (will each be 1 unless
      // --weights option supplied.
      Vector<BaseFloat> weights;
      if (weights_rspecifier != "") { // We have per-frame weighting.
        if (!weights_reader.HasKey(key)) {
          KALDI_WARN << "No per-frame weights available for utterance " << key;
          num_err++;
          continue;
        }
        weights = weights_reader.Value(key);
        if (weights.Dim() != file_frames) {
          KALDI_WARN << "Weights for utterance " << key << " have wrong dim "
                     << weights.Dim() << " vs. " << file_frames;
          num_err++;
          continue;
        }
      }
      if (gselect_rspecifier != "") {
        if (!gselect_reader.HasKey(key)) {
          KALDI_WARN << "No gselect information for utterance " << key;
          num_err++;
          continue;
        }
        const std::vector<std::vector<int32> > &gselect =
            gselect_reader.Value(key);
        if (gselect.size() != static_cast<size_t>(file_frames)) {
          KALDI_WARN << "gselect information for utterance " << key
                     << " has wrong size " << gselect.size() << " vs. "
                     << file_frames;
          num_err++;
          continue;
        }
        
        for (int32 i = 0; i < file_frames; i++) {
          BaseFloat weight = (weights.Dim() != 0) ? weights(i) : 1.0;
          if (weight == 0.0) continue;
          file_weight += weight;
          SubVector<BaseFloat> data1(mat1, i), data2(mat2, i);
          const std::vector<int32> &this_gselect = gselect[i];
          int32 gselect_size = this_gselect.size();
          KALDI_ASSERT(gselect_size > 0);
          Vector<BaseFloat> loglikes;
          gmm.LogLikelihoodsPreselect(data1, this_gselect, &loglikes);
          file_like += weight * loglikes.ApplySoftMax();
          loglikes.Scale(weight);
          for (int32 j = 0; j < loglikes.Dim(); j++)
            gmm_accs.AccumulateForComponent(data2, this_gselect[j], loglikes(j));
        }
      } else { // no gselect..
        Vector<BaseFloat> posteriors;
        for (int32 i = 0; i < file_frames; i++) {
          BaseFloat weight = (weights.Dim() != 0) ? weights(i) : 1.0;
          if (weight == 0.0) continue;
          file_weight += weight;
          file_like += weight * gmm.ComponentPosteriors(mat1.Row(i), &posteriors);
          posteriors.Scale(weight);
          gmm_accs.AccumulateFromPosteriors(mat2.Row(i), posteriors);
        }
      }
      KALDI_VLOG(2) << "File '" << key << "': Average likelihood = "
                    << (file_like/file_weight) << " over "
                    << file_weight <<" frames.";
      tot_like += file_like;
      tot_weight += file_weight;
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " files; "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per "
              << "frame = " << (tot_like/tot_weight) << " over " << tot_weight
              << " (weighted) frames.";

    WriteKaldiObject(gmm_accs, accs_wxfilename, binary);
    KALDI_LOG << "Written accs to " << accs_wxfilename;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

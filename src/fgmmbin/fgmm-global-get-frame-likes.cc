// fgmmbin/fgmm-global-get-frame-likes.cc

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
        "Print out per-frame log-likelihoods for each utterance, as an archive\n"
        "of vectors of floats.  If --average=true, prints out the average per-frame\n"
        "log-likelihood for each utterance, as a single float.\n"
        "Usage:  fgmm-global-get-frame-likes [options] <model-in> <feature-rspecifier> "
        "<likes-out-wspecifier>\n"
        "e.g.: fgmm-global-get-frame-likes 1.mdl scp:train.scp ark:1.likes\n";

    ParseOptions po(usage);
    bool average = false;
    std::string gselect_rspecifier;
    po.Register("gselect", &gselect_rspecifier, "rspecifier for gselect objects "
                "to limit the #Gaussians accessed on each frame.");
    po.Register("average", &average, "If true, print out the average per-frame "
                "log-likelihood as a single float per utterance.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        likes_wspecifier = po.GetArg(3);

    FullGmm fgmm;
    {
      bool binary_read;
      Input ki(model_filename, &binary_read);
      fgmm.Read(ki.Stream(), binary_read);
    }

    double tot_like = 0.0, tot_frames = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    BaseFloatVectorWriter likes_writer(average ? "" : likes_wspecifier);
    BaseFloatWriter average_likes_writer(average ? likes_wspecifier : "");
    int32 num_done = 0, num_err = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      int32 file_frames = mat.NumRows();
      Vector<BaseFloat> likes(file_frames);
      
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
          SubVector<BaseFloat> data(mat, i);
          const std::vector<int32> &this_gselect = gselect[i];
          int32 gselect_size = this_gselect.size();
          KALDI_ASSERT(gselect_size > 0);
          Vector<BaseFloat> loglikes;
          fgmm.LogLikelihoodsPreselect(data, this_gselect, &loglikes);
          likes(i) = loglikes.LogSumExp();
        }
      } else { // no gselect..
        for (int32 i = 0; i < file_frames; i++)
          likes(i) = fgmm.LogLikelihood(mat.Row(i));
      }

      tot_like += likes.Sum();
      tot_frames += file_frames;
      if (average)
        average_likes_writer.Write(key, likes.Sum() / file_frames);
      else
        likes_writer.Write(key, likes);
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " files; " << num_err
              << " with errors.";
    KALDI_LOG << "Overall likelihood per "
              << "frame = " << (tot_like/tot_frames) << " over " << tot_frames
              << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

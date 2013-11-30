// gmmbin/gmm-global-init-from-feats.cc

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {

// We initialize the GMM parameters by setting the variance to the global
// variance of the features, and the means to distinct randomly chosen frames.
void InitGmmFromRandomFrames(const Matrix<BaseFloat> &feats, DiagGmm *gmm) {
  int32 num_gauss = gmm->NumGauss(), num_frames = feats.NumRows(),
      dim = feats.NumCols();
  KALDI_ASSERT(num_frames >= 10 * num_gauss && "Too few frames to train on");
  Vector<double> mean(dim), var(dim);
  for (int32 i = 0; i < num_frames; i++) {
    mean.AddVec(1.0 / num_frames, feats.Row(i));
    var.AddVec2(1.0 / num_frames, feats.Row(i));
  }
  var.AddVec2(-1.0, mean);
  if (var.Max() <= 0.0)
    KALDI_ERR << "Features do not have positive variance " << var;
  
  DiagGmmNormal gmm_normal(*gmm);

  std::set<int32> used_frames;
  for (int32 g = 0; g < num_gauss; g++) {
    int32 random_frame = RandInt(0, num_frames - 1);
    while (used_frames.count(random_frame != 0))
      random_frame = RandInt(0, num_frames - 1);
    used_frames.insert(random_frame);
    gmm_normal.weights_(g) = 1.0 / num_gauss;
    gmm_normal.means_.Row(g).CopyFromVec(feats.Row(random_frame));
    gmm_normal.vars_.Row(g).CopyFromVec(var);
  }
  gmm->CopyFromNormal(gmm_normal);
  gmm->ComputeGconsts();
}

void TrainOneIter(const Matrix<BaseFloat> &feats,
                  const MleDiagGmmOptions &gmm_opts,
                  int32 iter,
                  int32 num_threads,
                  DiagGmm *gmm) {
  AccumDiagGmm gmm_acc(*gmm, kGmmAll);

  Vector<BaseFloat> frame_weights(feats.NumRows(), kUndefined);
  frame_weights.Set(1.0);

  double tot_like;
  tot_like = gmm_acc.AccumulateFromDiagMultiThreaded(*gmm, feats, frame_weights,
                                                     num_threads);

  KALDI_LOG << "Likelihood per frame on iteration " << iter
            << " was " << (tot_like / feats.NumRows()) << " over "
            << feats.NumRows() << " frames.";
  
  BaseFloat objf_change, count;
  MleDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);

  KALDI_LOG << "Objective-function change on iteration " << iter << " was "
            << (objf_change / count) << " over " << count << " frames.";
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "This program initializes a single diagonal GMM and does multiple iterations of\n"
        "training from features stored in memory.\n"
        "Usage:  gmm-global-init-feats [options] <feature-rspecifier> <model-out>\n"
        "e.g.: gmm-global-init-feats scp:train.scp 1.mdl\n";

    ParseOptions po(usage);
    MleDiagGmmOptions gmm_opts;
    
    bool binary = true;
    int32 num_gauss = 100;
    int32 num_gauss_init = 0;
    int32 num_iters = 50;
    int32 num_frames = 200000;
    int32 srand_seed = 0;
    int32 num_threads = 4;
    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-gauss", &num_gauss, "Number of Gaussians in the model");
    po.Register("num-gauss-init", &num_gauss_init, "Number of Gaussians in "
                "the model initially (if nonzero and less than num_gauss, "
                "we'll do mixture splitting)");
    po.Register("num-iters", &num_iters, "Number of iterations of training");
    po.Register("num-frames", &num_frames, "Number of feature vectors to store in "
                "memory and train on (randomly chosen from the input features)");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("num-threads", &num_threads, "Number of threads used for "
                "statistics accumulation");
                
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);    
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        model_wxfilename = po.GetArg(2);
    
    Matrix<BaseFloat> feats;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);


    KALDI_ASSERT(num_frames > 0);
    
    int64 num_read = 0, dim = 0;

    KALDI_LOG << "Reading features (will keep " << num_frames << " frames.)";
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      const Matrix<BaseFloat>  &this_feats = feature_reader.Value();
      for (int32 t = 0; t < this_feats.NumRows(); t++) {
        num_read++;
        if (dim == 0) {
          dim = this_feats.NumCols();
          feats.Resize(num_frames, dim);
        } else if (this_feats.NumCols() != dim) {
          KALDI_ERR << "Features have inconsistent dims "
                    << this_feats.NumCols() << " vs. " << dim
                    << " (current utt is) " << feature_reader.Key();
        }
        if (num_read <= num_frames) {
          feats.Row(num_read - 1).CopyFromVec(this_feats.Row(t));
        } else {
          BaseFloat keep_prob = num_frames / static_cast<BaseFloat>(num_read);
          if (WithProb(keep_prob)) { // With probability "keep_prob"
            feats.Row(RandInt(0, num_frames - 1)).CopyFromVec(this_feats.Row(t));
          }
        }
      }
    }

    if (num_read < num_frames) {
      KALDI_WARN << "Number of frames read " << num_read << " was less than "
                 << "target number " << num_frames << ", using all we read.";
      feats.Resize(num_read, dim, kCopyData);
    } else {
      BaseFloat percent = num_frames * 100.0 / num_read;
      KALDI_LOG << "Kept " << num_frames << " out of " << num_read
                << " input frames = " << percent << "%.";
    }

    if (num_gauss_init <= 0 || num_gauss_init > num_gauss)
      num_gauss_init = num_gauss;
    
    DiagGmm gmm(num_gauss_init, dim);
    
    KALDI_LOG << "Initializing GMM means from random frames to "
              << num_gauss_init << " Gaussians.";
    InitGmmFromRandomFrames(feats, &gmm);

    // we'll increase the #Gaussians by splitting,
    // till halfway through training.
    int32 cur_num_gauss = num_gauss_init,
        gauss_inc = (num_gauss - num_gauss_init) / (num_iters / 2);
        
    for (int32 iter = 0; iter < num_iters; iter++) {
      TrainOneIter(feats, gmm_opts, iter, num_threads, &gmm);

      int32 next_num_gauss = std::min(num_gauss, cur_num_gauss + gauss_inc);
      if (next_num_gauss > gmm.NumGauss()) {
        KALDI_LOG << "Splitting to " << next_num_gauss << " Gaussians.";
        gmm.Split(next_num_gauss, 0.1);
        cur_num_gauss = next_num_gauss;
      }
    }

    WriteKaldiObject(gmm, model_wxfilename, binary);
    KALDI_LOG << "Wrote model to " << model_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

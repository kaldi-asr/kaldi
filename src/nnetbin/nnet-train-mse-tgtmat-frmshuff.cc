// nnetbin/nnet-train-mse-tgtmat-frmshuff.cc

// Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)

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

/*
 * Karel : This tool has never been used, so there may be bugs.
 */

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by stochastic gradient descent.\n"
        "Usage:  nnet-train-mse-tgtmat-frmshuff [options] <model-in> <feature-rspecifier> <targets-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-mse-tgtmat-frmshuff nnet.init scp:train.scp ark:targets.scp nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 bunchsize=512, cachesize=32768, seed=777;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling (max 8388479)");
    po.Register("seed", &seed, "Seed value for srand, sets fixed order of frame-shuffling");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        targets_rspecifier = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    //set the seed to the pre-defined value
    srand(seed);
     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);

    CacheTgtMat cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Mse mse;

    
    CuMatrix<BaseFloat> feats, feats_transf, targets, nnet_in, nnet_out, nnet_tgt, obj_diff;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache, 
      // both reader are sequential, not to be too memory hungry,
      // the scp lists must be in the same order 
      // we run the loop over targets, skipping features with no targets
      while (!cache.Full() && !targets_reader.Done()) {
        // get the utts
        std::string tgt_utt = targets_reader.Key();
        std::string fea_utt = feature_reader.Key();
        // skip feature matrix with no targets
        while (fea_utt != tgt_utt) {
          KALDI_WARN << "No targets for: " << fea_utt;
          num_no_tgt_mat++;
          if (!feature_reader.Done()) {
            feature_reader.Next(); 
            fea_utt = feature_reader.Key();
          }
        }
        // now we should have a pair
        if (fea_utt == tgt_utt) {
          // get feature tgt_mat pair
          const Matrix<BaseFloat> &fea_mat = feature_reader.Value();
          const Matrix<BaseFloat> &tgt_mat = targets_reader.Value();
          // chech for dimension
          if (tgt_mat.NumRows() != fea_mat.NumRows()) {
            KALDI_WARN << "Alignment has wrong size "<< (tgt_mat.NumRows()) << " vs. "<< (fea_mat.NumRows());
            num_other_error++;
            continue;
          }
          // push features/targets to GPU
          feats = fea_mat;
          targets = tgt_mat;
          // possibly apply feature transform
          nnet_transf.Feedforward(feats, &feats_transf);
          // add to cache
          cache.AddData(feats_transf, targets);
          num_done++;
        }
        Timer t_features;
        feature_reader.Next(); 
        targets_reader.Next(); 
        time_next += t_features.Elapsed();

        //report the speed
        if (num_done % 1000 == 0) {
          time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }

      }
      // randomize
      if (!crossvalidate && randomize) {
        cache.Randomize();
      }
      // report
      KALDI_VLOG(1) << "Cache #" << ++num_cache << " "
                << (cache.Randomized()?"[RND]":"[NO-RND]")
                << " segments: " << num_done
                << " frames: " << static_cast<double>(total_frames)/360000 << "h";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&nnet_in, &nnet_tgt);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);
        mse.Eval(nnet_out, nnet_tgt, &obj_diff);
        if (!crossvalidate) {
          nnet.Backpropagate(obj_diff, NULL);
        }
        total_frames += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done()) break;
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }
    
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << time.Elapsed()/60 << "min, fps" << total_frames/time.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors.";

    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

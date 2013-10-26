// nnetbin/rbm-train-cd1-frmshuff.cc

// Copyright 2012  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-rbm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Train RBM by Contrastive Divergence alg. with 1 step of "
        "Markov Chain Monte-Carlo.\n"
        "The tool can perform several iterations (--num-iters) "
        "or it can subsample the training dataset (--drop-data)\n"
        "Usage:  rbm-train-cd1-frmshuff [options] <model-in> <feature-rspecifier> <model-out>\n"
        "e.g.: \n"
        " rbm-train-cd1-frmshuff 1.rbm.init scp:train.scp 1.rbm\n";

    ParseOptions po(usage);

    RbmTrainOptions trn_opts, trn_opts_rbm;
    trn_opts.Register(&po);

    bool binary = false; 
    po.Register("binary", &binary, "Write output in binary mode");

    int32 num_iters = 1; 
    po.Register("num-iters", &num_iters, 
                "Number of iterations (smaller datasets should have more iterations, "
                "iterating within tool becase of linear momentum scheduling)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 bunchsize=100, cachesize=32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling (max 8388479)");
    
    BaseFloat drop_data = 0.0; 
    po.Register("drop-data", &drop_data, "Threshold for random dropping of the data (0 no-drop, 1 drop-all)");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2);
        
    std::string target_model_filename;
    target_model_filename = po.GetArg(3);

     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet rbm_transf;
    if(feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.NumComponents()==1);
    KALDI_ASSERT(nnet.GetComponent(0).GetType() == Component::kRbm);
    RbmBase &rbm = dynamic_cast<RbmBase&>(nnet.GetComponent(0));

    // Configure the RBM
    // first get make some options easy to access:
    const BaseFloat& learn_rate = trn_opts.learn_rate;
    const BaseFloat& momentum = trn_opts.momentum;
    const BaseFloat& momentum_max = trn_opts.momentum_max;
    const int32& momentum_steps = trn_opts.momentum_steps;
    const int32& momentum_step_period = trn_opts.momentum_step_period;
    // trn_opts_rbm is to be passed the RBM, copy the opts
    trn_opts_rbm = trn_opts;
    trn_opts_rbm.learn_rate = learn_rate*(1-momentum);
    // pass options to RBM
    rbm.SetRbmTrainOptions(trn_opts_rbm);


    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Cache cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    CuRand<BaseFloat> cu_rand;
    MseProgress mse;

    
    CuMatrix<BaseFloat> feats, feats_transf, 
                        pos_vis, pos_hid, pos_hid_aux, 
                        neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;
    std::vector<int32> dummy_cache_vec;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << "RBM TRAINING STARTED";

    int32 iter = 1;
    KALDI_LOG << "Iteration " << iter << "/" << num_iters;

    int32 num_done = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !feature_reader.Done()) {
        // get feature matrix
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        // push features to GPU
        feats.Resize(mat.NumRows(),mat.NumCols());
        feats.CopyFromMat(mat);
        // possibly apply transform (may contain splicing)
        rbm_transf.Feedforward(feats, &feats_transf);
        // subsample training data to get faster epochs on large datasets
        if(drop_data > 0.0) {
          Matrix<BaseFloat> mat2(feats_transf.NumRows(), feats_transf.NumCols(),
                                 kUndefined);
          feats_transf.CopyToMat(&mat2);
          for(int32 r=mat2.NumRows()-1; r >= 0; r--) {
            if(RandUniform() < drop_data) {
              mat2.RemoveRow(r);
            }
          }
          if(mat2.NumRows() == 0) continue;
          feats_transf.Resize(mat2.NumRows(),mat2.NumCols());
          feats_transf.CopyFromMat(mat2);
        }
        // resize the dummy vector to fill Cache:: with
        dummy_cache_vec.resize(feats_transf.NumRows());
        // add to cache
        cache.AddData(feats_transf, dummy_cache_vec);
        num_done++;
        // next feature file... 
        Timer t_features;
        feature_reader.Next(); 
        time_next += t_features.Elapsed();

        // report the speed
        if (num_done % 5000 == 0) {
          time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }
      // randomize
      cache.Randomize();
      // report
      KALDI_VLOG(2) << "Cache #" << ++num_cache << " "
                << (cache.Randomized()?"[RND]":"[NO-RND]")
                << " segments: " << num_done
                << " frames: " << static_cast<double>(total_frames)/360000 << "h";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&pos_vis, &dummy_cache_vec);
        // get the dims 
        int32 num_frames = pos_vis.NumRows(),
              dim_hid = rbm.OutputDim();
       
        // TRAIN with CD1
        // forward pass
        rbm.Propagate(pos_vis, &pos_hid);

        // alter the hidden values, so we can generate negative example
        if (rbm.HidType() == Rbm::BERNOULLI) {
          pos_hid_aux.Resize(num_frames, dim_hid);
          cu_rand.BinarizeProbs(pos_hid, &pos_hid_aux);
        } else {
          // assume HidType Rbm::GAUSSIAN
          pos_hid_aux.Resize(num_frames, dim_hid);
          pos_hid_aux.CopyFromMat(pos_hid);
          cu_rand.AddGaussNoise(&pos_hid_aux);
        }

        // reconstruct pass
        rbm.Reconstruct(pos_hid_aux, &neg_vis);
        // propagate negative examples
        rbm.Propagate(neg_vis, &neg_hid);
        // update step
        rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
        // evaluate mean square error
        mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

        total_frames += num_frames;

        // change the momentum progressively per 0.5million samples of the data
        {
          static int32 n_prev = -1;
          BaseFloat step = (momentum_max - momentum) / momentum_steps;
          int32 n = total_frames / momentum_step_period; //change every momentum_step_period data
          BaseFloat momentum_actual;
          if(n > momentum_steps) {
            momentum_actual = momentum_max;
          } else {
            momentum_actual = momentum + n*step;
          }
          if(n - n_prev > 0) {
            n_prev = n;
            BaseFloat learning_rate_actual = learn_rate*(1-momentum_actual);
            KALDI_VLOG(1) << "Setting momentum " << momentum_actual 
                          << " and learning rate " << learning_rate_actual
                          << " after processing " 
                          << static_cast<double>(total_frames)/360000 << "h";
            // pass values to rbm
            trn_opts_rbm.momentum = momentum_actual;
            trn_opts_rbm.learn_rate = learning_rate_actual;
            rbm.SetRbmTrainOptions(trn_opts_rbm);
          }
        }
      }

      // reopen the feature stream if we will run another iteration
      if (feature_reader.Done() && (iter < num_iters)) {
        iter++;
        KALDI_LOG << "Iteration " << iter << "/" << num_iters;
        feature_reader.Close();
        feature_reader.Open(feature_rspecifier);
      }
        
      // otherwise stop the training
      if (feature_reader.Done()) break;
    }

    nnet.Write(target_model_filename, binary);
    
    KALDI_LOG << "RBM TRAINING FINISHED " 
              << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << iter << " iterations, " << num_done << " files.";

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

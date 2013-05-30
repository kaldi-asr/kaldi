// nnetbin/rbm-train-cd1-frmshuff.cc

// Copyright 2012  Brno University of Technology (Author: Karel Vesely)

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
  try {
    const char *usage =
        "Train an RBM by contrastive divergence alg.\n"
        "Usage:  rbm-train-cd1-frmshuff [options] <model-in> <feature-rspecifier> <model-out>\n"
        "e.g.: \n"
        " rbm-train-cd1-frmshuff 1.rbm.init scp:train.scp 1.rbm\n";

    ParseOptions po(usage);
    bool binary = false; 
    po.Register("binary", &binary, "Write output in binary mode");

    int32 num_iters = 1; 
    po.Register("num-iters", &num_iters, "Number of iterations (smaller datasets should be seen more than once, iterating within tool becase of linear momentum scheduling)");
 
    BaseFloat learn_rate = 0.01,
        momentum = 0.5,
        momentum_max = 0.9,
        l2_penalty = 0.0002;
     po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("momentum-max", &momentum_max, "Momentum maximal");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    
    /**
     * adding linear momentum scheduling
     */
    int32 momentum_steps = 40,
        momentum_step_period = 500000; //500000 * 40 = 55h of linear increase of momentum
    po.Register("momentum-steps", &momentum_steps, "Number of steps of linear momentum scheduling");
    po.Register("momentum-step-period", &momentum_step_period, "Number of datapoints per single momentum increase step");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 bunchsize=100, cachesize=32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling (max 8388479)");
    
    BaseFloat drop_data = 0.0; 
    po.Register("drop-data", &drop_data, "Threshold for random dropping of the data (0 no-drop, 1 drop-all)");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2 ;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#endif

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
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet rbm_transf;
    if(feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRbm);
    RbmBase &rbm = dynamic_cast<RbmBase&>(*nnet.Layer(0));

    rbm.SetLearnRate(learn_rate*(1-momentum));
    rbm.SetMomentum(momentum);
    rbm.SetL2Penalty(l2_penalty);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Cache cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    CuRand<BaseFloat> cu_rand;
    MseProgress mse;

    
    CuMatrix<BaseFloat> feats, feats_transf, pos_vis, pos_hid, neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;
    std::vector<int32> dummy_cache_vec;

    Timer tim;
    double time_next=0;
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
        // subsample the feats to get faster epochs
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
      }
      // randomize
      cache.Randomize();
      // report
      KALDI_VLOG(1) << "Cache #" << ++num_cache << " "
                << (cache.Randomized()?"[RND]":"[NO-RND]")
                << " segments: " << num_done
                << " frames: " << static_cast<double>(tot_t)/360000 << "h";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&pos_vis, &dummy_cache_vec);
       
        // TRAIN with CD1
        // forward pass
        rbm.Propagate(pos_vis, &pos_hid);

        // alter the hidden values, so we can generate negative example
        if (rbm.HidType() == Rbm::BERNOULLI) {
          neg_hid.Resize(pos_hid.NumRows(),pos_hid.NumCols());
          cu_rand.BinarizeProbs(pos_hid, &neg_hid);
        } else {
          // assume Rbm::GAUSSIAN
          neg_hid.Resize(pos_hid.NumRows(),pos_hid.NumCols());
          neg_hid.CopyFromMat(pos_hid);
          cu_rand.AddGaussNoise(&neg_hid);
        }

        // reconstruct pass
        rbm.Reconstruct(neg_hid, &neg_vis);
        // propagate negative examples
        rbm.Propagate(neg_vis, &neg_hid);
        // update step
        rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
        // evaluate mean square error
        mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

        tot_t += pos_vis.NumRows();

        // change the momentum progressively per 0.5million samples of the data
        {
          static int32 n_prev = -1;
          BaseFloat step = (momentum_max - momentum) / momentum_steps;
          int32 n = tot_t / momentum_step_period; //change every momentum_step_period data
          BaseFloat momentum_actual;
          if(n > momentum_steps) {
            momentum_actual = momentum_max;
          } else {
            momentum_actual = momentum + n*step;
          }
          if(n - n_prev > 0) {
            n_prev = n;
            BaseFloat learning_rate_actual = learn_rate*(1-momentum_actual);
            KALDI_LOG << "Setting momentum : " << momentum_actual 
                      << " learning rate : " << learning_rate_actual;
            rbm.SetMomentum(momentum_max);
            rbm.SetLearnRate(learning_rate_actual);
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
              << tim.Elapsed()/60 << " min, fps" << tot_t/tim.Elapsed()
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

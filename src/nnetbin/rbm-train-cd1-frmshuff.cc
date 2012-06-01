// rbmbin/rbm-train-cd1-frmshuff.cc

// Copyright 2012  Karel Vesely

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
        "Perform iteration of RBM training by contrastive divergence alg.\n"
        "Usage:  rbm-train-cd1-frmshuff [options] <model-in> <feature-rspecifier> <model-out>\n"
        "e.g.: \n"
        " rbm-train-cd1-frmshuff rbm.init scp:train.scp rbm.iter1\n";

    ParseOptions po(usage);
    bool binary = false; 
    po.Register("binary", &binary, "Write output in binary mode");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    int32 bunchsize=512, cachesize=32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling");

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


    Nnet rbm_transf;
    if(feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRbm);
    RbmBase& rbm = dynamic_cast<RbmBase&>(*nnet.Layer(0));

    rbm.SetLearnRate(learn_rate);
    rbm.SetMomentum(momentum);
    rbm.SetL2Penalty(l2_penalty);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Cache cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    CuRand<BaseFloat> cu_rand;
    Mse mse;

    
    CuMatrix<BaseFloat> feats, feats_transf, pos_vis, pos_hid, neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;
    std::vector<int32> dummy_cache_vec;

    Timer tim;
    double time_next=0;
    KALDI_LOG << "RBM TRAINING STARTED";

    int32 num_done = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !feature_reader.Done()) {
        // get feature matrix
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        // resize the dummy vector to fill Cache:: with
        dummy_cache_vec.resize(mat.NumRows());
        // push features to GPU
        feats.CopyFromMat(mat);
        // possibly apply transform
        rbm_transf.Feedforward(feats, &feats_transf);
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
      std::cerr << "Cache #" << ++num_cache << " "
                << (cache.Randomized()?"[RND]":"[NO-RND]")
                << " segments: " << num_done
                << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&pos_vis, &dummy_cache_vec);
       
        // TRAIN with CD1
        // forward pass
        rbm.Propagate(pos_vis, &pos_hid);
        // alter the hidden values, so we can generate negative example
        if (rbm.HidType() == Rbm::BERNOULLI) {
          cu_rand.BinarizeProbs(pos_hid, &neg_hid);
        } else {
          // assume Rbm::GAUSSIAN
          neg_hid.CopyFromMat(pos_hid);
          cu_rand.AddGaussNoise(&neg_hid);
        }
        // reconstruct pass
        rbm.Reconstruct(neg_hid, &pos_vis);
        // update step
        rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
        // evaluate mean square error
        mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

        tot_t += pos_vis.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done()) break;
    }

    nnet.Write(target_model_filename, binary);
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << "RBM TRAINING FINISHED " 
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files.";

    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

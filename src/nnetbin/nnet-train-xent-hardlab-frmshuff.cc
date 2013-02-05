// nnetbin/nnet-train-xent-hardlab-perutt.cc

// Copyright 2011  Karel Vesely

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

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
        "Usage:  nnet-train-xent-hardlab-frmshuff [options] <model-in> <feature-rspecifier> <alignments-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-xent-hardlab-perutt nnet.init scp:train.scp ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);
    bool binary = false, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    int32 bunchsize=512, cachesize=32768, seed=777;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling");
    po.Register("seed", &seed, "Seed value for srand, sets fixed order of frame-shuffling");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#endif
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    //set the seed to the pre-defined value
    srand(seed);
     
    using namespace kaldi;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    if(use_gpu_id > -2)
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    nnet.SetLearnRate(learn_rate, NULL);
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    Cache cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, obj_diff;
    std::vector<int32> targets;

    Timer tim;
    double time_next=0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !feature_reader.Done()) {
        std::string key = feature_reader.Key();
        if (!alignments_reader.HasKey(key)) {
          num_no_alignment++;
        } else {
          // get feature alignment pair
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          const std::vector<int32> &alignment = alignments_reader.Value(key);
          // chech for dimension
          if ((int32)alignment.size() != mat.NumRows()) {
            KALDI_WARN << "Alignment has wrong size, ali "<< (alignment.size()) << " vs. feats "<< (mat.NumRows()) << ", " << key;
            num_other_error++;
          } else { //dimension OK
            // push features to GPU
            feats.Resize(mat.NumRows(), mat.NumCols(), kUndefined);
            feats.CopyFromMat(mat);
            // possibly apply transform
            nnet_transf.Feedforward(feats, &feats_transf);
            // add to cache
            cache.AddData(feats_transf, alignment);
            num_done++;
          }
        }
        Timer t_features;
        feature_reader.Next(); 
        time_next += t_features.Elapsed();
      }
      // randomize
      if (!crossvalidate && randomize) {
        cache.Randomize();
      }
      // report
      std::cerr << "Cache #" << ++num_cache << " "
                << (cache.Randomized()?"[RND]":"[NO-RND]")
                << " segments: " << num_done
                << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&nnet_in, &targets);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);
        xent.EvalVec(nnet_out, targets, &obj_diff);
        if (!crossvalidate) {
          nnet.Backpropagate(obj_diff, NULL);
        }
        tot_t += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done()) break;
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

    KALDI_LOG << xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

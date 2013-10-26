// nnetbin/nnet-train-xent-hardlab-frmshuff.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by stochastic gradient descent.\n"
        "Usage:  nnet-train-xent-hardlab-frmshuff [options] <model-in> <feature-rspecifier> <alignments-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-xent-hardlab-perutt nnet.init scp:train.scp ark:train.ali nnet.iter1\n";

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
    
    kaldi::int32 max_frames = 6000; // Allow segments maximum of one minute by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    
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
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    Cache cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, obj_diff;
    std::vector<int32> targets;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !feature_reader.Done()) {
        std::string utt = feature_reader.Key();
        KALDI_VLOG(2) << "Reading utt " << utt;
        // check that we have alignments
        if (!alignments_reader.HasKey(utt)) {
          num_no_alignment++;
          feature_reader.Next(); 
          continue;
        }
        // get feature alignment pair
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(utt);
        // check maximum length of utterance
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Utterance " << utt << ": Skipped because it has " << mat.NumRows() << 
            " frames, which is more than " << max_frames << ".";
          num_other_error++;
          feature_reader.Next(); 
          continue;
        }
        // check length match of features/alignments
        if ((int32)alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignment has wrong length, ali " << (alignment.size()) 
                     << " vs. feats "<< (mat.NumRows()) << ", utt " << utt;
          num_other_error++;
          feature_reader.Next();
          continue;
        }
        
        // All the checks OK,
        // push features to GPU
        feats.Resize(mat.NumRows(), mat.NumCols(), kUndefined);
        feats.CopyFromMat(mat);
        // possibly apply transform
        nnet_transf.Feedforward(feats, &feats_transf);
        // add to cache
        cache.AddData(feats_transf, alignment);
        num_done++;
        
        // measure the time needed to get next feature file 
        Timer t_features;
        feature_reader.Next(); 
        time_next += t_features.Elapsed();
        // report the speed
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
        cache.GetBunch(&nnet_in, &targets);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);
        xent.EvalVec(nnet_out, targets, &obj_diff);
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

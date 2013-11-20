// nnetbin/nnet-train-xent-hardlab-perutt.cc

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
        "Usage:  nnet-train-xent-hardlab-perutt [options] <model-in> <feature-rspecifier> <alignments-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-xent-hardlab-perutt nnet.init scp:train.scp ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    // Add dummy options, to make the tool compatible with standard scripts
    int32 bunchsize, cachesize, seed;
    bool randomize;
    po.Register("bunchsize", &bunchsize, "Unused...");
    po.Register("cachesize", &cachesize, "Unused...");
    po.Register("seed", &seed, "Unused...");
    po.Register("randomize", &randomize, "Unused...");
    //
    kaldi::int32 max_frames = 6000; // Allow segments maximum of one minute by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#else
    int32 use_gpu_id=0;
    po.Register("use-gpu-id", &use_gpu_id, "Unused, kaldi is compiled w/o CUDA");
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
     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
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

    Xent xent;
    
    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, obj_diff;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
    while (!feature_reader.Done()) {
      std::string utt = feature_reader.Key();
      // check that we have alignments
      if (!alignments_reader.HasKey(utt)) {
        num_no_alignment++;
        feature_reader.Next(); 
        continue;
      } 
      // get feature alignment pair  
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::vector<int32> alignment = alignments_reader.Value(utt);
      // check maximum length of utterance
      if (mat.NumRows() > max_frames) {
        KALDI_WARN << "Utterance " << utt << ": Skipped because it has " 
          << mat.NumRows() << " frames, which is more than " << max_frames << ".";
        num_other_error++;
        feature_reader.Next(); 
        continue;
      }
      // check length match of features/alignments
      if ((int32)alignment.size() != mat.NumRows()) {
        int32 diff = alignment.size() -  mat.NumRows();
        int32 tolerance = 5; // allow some tolerance (truncate)
        if (diff > 0 && diff < tolerance) { // alignment longer
          for(int32 i=0; i<diff; i++) { alignment.pop_back(); }
        }
        if (diff < 0 && abs(diff) < tolerance) { // feature matrix longer
          for(int32 i=0; i<abs(diff); i++) { mat.RemoveRow(mat.NumRows()-1); }
        }
        // check again
        if ((int32)alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Length mismatch of alignment "<< (alignment.size()) << " vs. features "<< (mat.NumRows());
          num_other_error++;
          feature_reader.Next();
          continue;
        }
      }
      // check max value in alignment corresponds to NN output
      KALDI_ASSERT(*std::max_element(alignment.begin(),alignment.end()) < nnet.OutputDim());

      // log
      KALDI_VLOG(2) << "utt " << utt << ", frames " << alignment.size();

      // push features to GPU
      feats.Resize(mat.NumRows(),mat.NumCols());
      feats.CopyFromMat(mat);

      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Propagate(feats_transf, &nnet_out);
        
      xent.EvalVec(nnet_out, alignment, &obj_diff);
        
      if (!crossvalidate) {
        nnet.Backpropagate(obj_diff, NULL);
      }

      total_frames += mat.NumRows();
    
      num_done++;
      Timer t_features;
      feature_reader.Next();
      time_next += t_features.Elapsed();

      // monitor the training
      if((kaldi::g_kaldi_verbose_level >= 3) && (num_done % 100 == 0)) {
        if(!crossvalidate) {
          KALDI_VLOG(3) << nnet.InfoGradient();
        } else {
          KALDI_VLOG(3) << nnet.InfoPropagate();
        }
      }

      // check for nan/inf in network weights
      if (num_done % 1000 == 0) {
        nnet.Check();
      }

      // report the speed
      if (num_done % 1000 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << total_frames/time_now
                      << " frames per second.";
      }
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

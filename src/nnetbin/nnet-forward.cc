// nnetbin/nnet-forward.cc

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

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "Usage:  nnet-forward [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-forward nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts, "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register("prior-scale", &prior_scale, "scaling factor of prior log-probabilites given by --class-frame-counts");

    bool apply_log = false, silent = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    po.Register("silent", &silent, "Don't print any messages");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
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
    //optionally remove softmax
    if(no_softmax && nnet.Layer(nnet.LayerCount()-1)->GetType() == Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveLayer(nnet.LayerCount()-1);
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    // Read the class-counts, compute priors
    Vector<BaseFloat> tmp_priors;
    CuVector<BaseFloat> priors;
    if(class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();
     
      //create inv. priors, or log inv priors 
      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0/sum);
      if (apply_log || no_softmax) {
        tmp_priors.ApplyLog();
        tmp_priors.Scale(-prior_scale);
      } else {
        tmp_priors.ApplyPow(-prior_scale);
      }

      //detect the inf, replace by something reasonable (maximal non-inf value)
      //a) replace inf by -inf
      for(int32 i=0; i<tmp_priors.Dim(); i++) {
        if(tmp_priors(i) == std::numeric_limits<BaseFloat>::infinity()) {
          tmp_priors(i) *= -1.0;
        }
      }
      //b) find max
      BaseFloat max = tmp_priors.Max();
      //c) replace -inf by max prior
      for(int32 i=0; i<tmp_priors.Dim(); i++) {
        if(tmp_priors(i) == -std::numeric_limits<BaseFloat>::infinity()) {
          tmp_priors(i) = max;
        }
      }

      // push priors to GPU
      priors.Resize(tmp_priors.Dim());
      priors.CopyFromVec(tmp_priors);
    }

    Timer tim;
    if(!silent) KALDI_LOG << "MLP FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      //check for NaN/inf
      for(int32 r=0; r<mat.NumRows(); r++) {
        for(int32 c=0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          if(val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }
      // push it to gpu
      feats = mat;
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);
      
      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }
     
      // divide posteriors by priors to get quasi-likelihoods
      if(class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out.MulColsVec(priors);
        }
      }
     
      //download from GPU
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for(int32 r=0; r<nnet_out_host.NumRows(); r++) {
        for(int32 c=0; c<nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r,c);
          if(val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
        }
      }
      // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 1000 == 0) {
        if(!silent) KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    if(!silent) KALDI_LOG << "MLP FEEDFORWARD FINISHED " 
                          << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed(); 
    if(!silent) KALDI_LOG << "Done " << num_done << " files";

#if HAVE_CUDA==1
    if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done>0)?0:1);
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

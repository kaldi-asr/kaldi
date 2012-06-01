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

#include "nnet_cpu/nnet-nnet.h"
#include "nnet_cpu/nnet-loss.h"
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

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

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

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    Matrix<BaseFloat> feats_transf, nnet_out;

    // Read the class-counts, compute priors
    Vector<BaseFloat> priors;
    if(class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      priors.Read(in.Stream(), false);
      in.Close();
      
      BaseFloat sum = priors.Sum();
      priors.Scale(1.0/sum);
      if (apply_log) {
        priors.ApplyLog();
        priors.Scale(-1.0);
      } else {
        priors.InvertElements();
      }
    }

    Timer tim;
    KALDI_LOG << "MLP FEEDFORWARD STARTED";
    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      // fwd-pass
      nnet_transf.Feedforward(mat, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);
      
      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }
     
      // divide posteriors by priors to get quasi-likelihoods
      if(class_frame_counts != "") {
        if (apply_log) {
          for(int32 r=0; r<nnet_out.NumRows(); r++) {
            nnet_out.Row(r).AddVec(1.0, priors);
          } 
        } else {
          nnet_out.MulColsVec(priors);
        }
      }
 
      // write
      feature_writer.Write(feature_reader.Key(), nnet_out);

      // progress log
      if (num_done % 1000 == 0) {
        KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "MLP FEEDFORWARD FINISHED " 
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed(); 
    KALDI_LOG << "Done " << num_done << " files";

    return 0;
  } catch(const std::exception& e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

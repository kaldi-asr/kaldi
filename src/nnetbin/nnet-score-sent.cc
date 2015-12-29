// nnetbin/nnet-score-sent.cc

// Copyright 2014 The Johns Hopkins University (Author: Sri Harish Mallidi)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform forward pass through Neural Network and output sentence level scores.\n"
        "\n"
        "Usage:  nnet-score-sent [options] <model-in> <feature-rspecifier> <targets-rspecifier> <scores-wspecifier>\n"
        "e.g.: \n"
        " nnet-score-sent nnet ark:features.ark ark:posteriors.ark ark:scores.xent\n";

    ParseOptions po(usage);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");
    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        targets_rspecifier = po.GetArg(3),
        scores_wspecifier = po.GetArg(4);
        
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

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    BaseFloatMatrixWriter scores_writer(scores_wspecifier);

    Xent xent;
    Mse mse;

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, obj_diff, out_mat;
    Matrix<BaseFloat> out_mat_host;
    Vector<BaseFloat> dummy_frame_weights;

    Timer time;
    double time_now = 0;
    int32 num_done = 0, num_other_error = 0;
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string utt = feature_reader.Key();
      Matrix<BaseFloat> mat = feature_reader.Value();
      Posterior nnet_tgt = targets_reader.Value(utt);
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << feature_reader.Key() 
                    << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      for(int32 r=0; r<mat.NumRows(); r++) {
        for(int32 c=0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          if(val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }
      // correct small length mismatch ... or drop sentence
      {
	// add lengths to vector
	std::vector<int32> lenght;
	lenght.push_back(mat.NumRows());
	lenght.push_back(nnet_tgt.size());
	// find min, max
	int32 min = *std::min_element(lenght.begin(),lenght.end());
	int32 max = *std::max_element(lenght.begin(),lenght.end());
	// fix or drop ?
	if (max - min < length_tolerance) {
	  if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
	  if(nnet_tgt.size() != min) nnet_tgt.resize(min);
	} else {
	  KALDI_WARN << utt << ", length mismatch of targets " << nnet_tgt.size()
		     << " and features " << mat.NumRows();
	  num_other_error++;
	  continue;
	}
      }
      
      // push it to gpu
      feats = mat;
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);

      // allocate dummy_frame_weights vector and set to 1
      dummy_frame_weights.Resize(nnet_out.NumRows(),  kSetZero);
      dummy_frame_weights.Add(1.0);

      // evaluate objective function we've chosen
      if (objective_function == "xent") {
	xent.Eval(dummy_frame_weights, nnet_out, nnet_tgt, &obj_diff);
      } else if (objective_function == "mse") {
	mse.Eval(dummy_frame_weights, nnet_out, nnet_tgt, &obj_diff);
	
	// write in output
	// compute framelevel squared error
	CuMatrix<BaseFloat> diff_pow_2 = obj_diff;
	diff_pow_2.MulElements(diff_pow_2); // (y)^2
	CuVector<BaseFloat> l2norm_diff;
	l2norm_diff.Resize(obj_diff.NumRows());
	l2norm_diff.AddColSumMat(1.0, diff_pow_2, 0.0); // sum over cols (pdfs)
	l2norm_diff.ApplyPow(0.5); // l2 norm 
	
	// add l2norm_diff to column of obj_diff
	out_mat.Resize(obj_diff.NumRows(), obj_diff.NumCols()+1);
	out_mat.CopyColFromVec(l2norm_diff, 0);
	CuSubMatrix<BaseFloat> out_mat_sub(out_mat.ColRange(1,obj_diff.NumCols()));
	out_mat_sub.CopyFromMat(obj_diff);

	out_mat_host.Resize(out_mat.NumRows(), out_mat.NumCols());
	out_mat.CopyToMat(&out_mat_host);

	// write
	scores_writer.Write(feature_reader.Key(), out_mat_host);

      } else {
	KALDI_ERR << "Unknown objective function code : " << objective_function;
      }

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

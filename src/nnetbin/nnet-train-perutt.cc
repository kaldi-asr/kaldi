// nnetbin/nnet-train-perutt.cc

// Copyright 2011-2014  Brno University of Technology (Author: Karel Vesely)

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
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
      "Perform one iteration of NN training by SGD with per-utterance updates.\n"
      "The training targets are represented as pdf-posteriors, usually prepared "
      "by ali-to-post.\n"
      "Usage: nnet-train-perutt [options] "
      "<feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
      "e.g.: nnet-train-perutt scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
        "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in Nnet format");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,
        "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
        "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
        "Per-frame weights to scale gradients (frame selection/weighting).");

    kaldi::int32 max_frames = 6000;  // Allow segments maximum of one minute by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    //// Add dummy option for compatibility with default scheduler,
    bool randomize = false;
    po.Register("randomize", &randomize,
        "Dummy, for compatibility with 'steps/nnet/train_scheduler.sh'");
    ////

    po.Read(argc, argv);

    if (po.NumArgs() != 3 + (crossvalidate ? 0 : 1)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    Xent xent;
    Mse mse;

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0,
          num_no_tgt_mat = 0,
          num_other_error = 0;

    // main loop,
    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      KALDI_VLOG(3) << "Reading " << utt;
      // check that we have targets
      if (!targets_reader.HasKey(utt)) {
        KALDI_WARN << utt << ", missing targets";
        num_no_tgt_mat++;
        continue;
      }
      // check we have per-frame weights
      if (frame_weights != "" && !weights_reader.HasKey(utt)) {
        KALDI_WARN << utt << ", missing per-frame weights";
        num_other_error++;
        feature_reader.Next();
        continue;
      }
      // get feature / target pair
      Matrix<BaseFloat> mat = feature_reader.Value();
      Posterior targets = targets_reader.Value(utt);
      // skip the sentence if it is too long,
      if (mat.NumRows() > max_frames) {
        KALDI_WARN << "Skipping " << utt
          << " that has " << mat.NumRows() << " frames,"
          << " it is longer than '--max-frames'" << max_frames;
        num_other_error++;
        continue;
      }
      // get per-frame weights
      Vector<BaseFloat> weights;
      if (frame_weights != "") {
        weights = weights_reader.Value(utt);
      } else {  // all per-frame weights are 1.0
        weights.Resize(mat.NumRows());
        weights.Set(1.0);
      }
      // correct small length mismatch ... or drop sentence
      {
        // add lengths to vector
        std::vector<int32> length;
        length.push_back(mat.NumRows());
        length.push_back(targets.size());
        length.push_back(weights.Dim());
        // find min, max
        int32 min = *std::min_element(length.begin(), length.end());
        int32 max = *std::max_element(length.begin(), length.end());
        // fix or drop ?
        if (max - min < length_tolerance) {
          if (mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
          if (targets.size() != min) targets.resize(min);
          if (weights.Dim() != min) weights.Resize(min, kCopyData);
        } else {
          KALDI_WARN << utt << ", length mismatch of targets " << targets.size()
                     << " and features " << mat.NumRows();
          num_other_error++;
          continue;
        }
      }
      // apply optional feature transform
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

      // forward pass
      nnet.Propagate(feats_transf, &nnet_out);

      // evaluate objective function we've chosen,
      if (objective_function == "xent") {
        // gradients are re-scaled by weights inside Eval,
        xent.Eval(weights, nnet_out, targets, &obj_diff);
      } else if (objective_function == "mse") {
        // gradients are re-scaled by weights inside Eval,
        mse.Eval(weights, nnet_out, targets, &obj_diff);
      } else {
        KALDI_ERR << "Unknown objective function code : "
                  << objective_function;
      }

      if (!crossvalidate) {
        // backpropagate and update,
        nnet.Backpropagate(obj_diff, NULL);
      }

      // 1st minibatch : show what happens in network,
      if (total_frames == 0) {
        KALDI_VLOG(1) << "### After " << total_frames << " frames,";
        KALDI_VLOG(1) << nnet.InfoPropagate();
        if (!crossvalidate) {
          KALDI_VLOG(1) << nnet.InfoBackPropagate();
          KALDI_VLOG(1) << nnet.InfoGradient();
        }
      }

      // VERBOSE LOG
      // monitor the NN training (--verbose=2),
      if (GetVerboseLevel() >= 2) {
        static int32 counter = 0;
        counter += mat.NumRows();
        // print every 25k frames,
        if (counter >= 25000) {
          KALDI_VLOG(2) << "### After " << total_frames << " frames,";
          KALDI_VLOG(2) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(2) << nnet.InfoBackPropagate();
            KALDI_VLOG(2) << nnet.InfoGradient();
          }
          counter = 0;
        }
      }

      num_done++;
      total_frames += weights.Sum();

      // do this every 5000 utterances,
      if (num_done % 5000 == 0) {
        // report the speed,
        double time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: "
          << "time elapsed = " << time_now / 60 << " min; "
          << "processed " << total_frames / time_now << " frames per sec.";
#if HAVE_CUDA == 1
        // check that GPU computes accurately,
        CuDevice::Instantiate().CheckGpuHealth();
#endif
      }
    }  // main loop,

    // after last minibatch : show what happens in network,
    KALDI_VLOG(1) << "### After " << total_frames << " frames,";
    KALDI_VLOG(1) << nnet.InfoPropagate();
    if (!crossvalidate) {
      KALDI_VLOG(1) << nnet.InfoBackPropagate();
      KALDI_VLOG(1) << nnet.InfoGradient();
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, "
      << num_no_tgt_mat << " with no tgt_mats, "
      << num_other_error << " with other errors. "
      << "[" << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
      << ", " << (randomize ? "RANDOMIZED" : "NOT-RANDOMIZED")
      << ", " << time.Elapsed() / 60 << " min, processing "
      << total_frames / time.Elapsed() << " frames per sec.]";

    if (objective_function == "xent") {
      KALDI_LOG << xent.ReportPerClass();
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

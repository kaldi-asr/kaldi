// nnetbin/nnet-train-frmshuff.cc

// Copyright 2013-2016  Brno University of Technology (Author: Karel Vesely)

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
      "Perform one iteration (epoch) of Neural Network training with\n"
      "mini-batch Stochastic Gradient Descent. The training targets\n"
      "are usually pdf-posteriors, prepared by ali-to-post.\n"
      "Usage:  nnet-train-frmshuff [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
      "e.g.: nnet-train-frmshuff scp:feats.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
        "Perform cross-validation (don't back-propagate)");

    bool randomize = true;
    po.Register("randomize", &randomize,
        "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in Nnet format");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,
        "Objective function : xent|mse|multitask");

    int32 max_frames = 360000;
    po.Register("max-frames", &max_frames,
        "Maximum number of frames an utterance can have (skipped if longer)");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
        "Allowed length mismatch of features/targets/weights "
        "(in frames, we truncate to the shortest)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
        "Per-frame weights, used to re-scale gradients.");

    std::string utt_weights;
    po.Register("utt-weights", &utt_weights,
        "Per-utterance weights, used to re-scale frame-weights.");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

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
    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent(loss_opts);
    Mse mse(loss_opts);

    MultiTaskLoss multitask(loss_opts);
    if (0 == objective_function.compare(0, 9, "multitask")) {
      // objective_function contains something like :
      // 'multitask,xent,2456,1.0,mse,440,0.001'
      //
      // the meaning is following:
      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
      multitask.InitFromString(objective_function);
    }

    CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;

    Timer time, time_io;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    int32 num_done = 0,
          num_no_tgt_mat = 0,
          num_other_error = 0;

    double time_io_accu = 0.0;

    // main loop,
    while (!feature_reader.Done()) {
#if HAVE_CUDA == 1
      // check that GPU computes accurately,
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer,
      time_io.Reset();
      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        if (feature_randomizer.IsFull()) {
          // break the loop without calling Next(),
          // we keep the 'utt' for next round,
          break;
        }
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // check that we have targets,
        if (!targets_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
        // check we have per-frame weights,
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // check we have per-utterance weights,
        if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-utterance weight";
          num_other_error++;
          continue;
        }
        // get feature / target pair,
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior targets = targets_reader.Value(utt);
        // get per-frame weights,
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else {  // all per-frame weights are 1.0,
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }
        // multiply with per-utterance weight,
        if (utt_weights != "") {
          BaseFloat w = utt_weights_reader.Value(utt);
          KALDI_ASSERT(w >= 0.0);
          if (w == 0.0) continue;  // remove sentence from training,
          weights.Scale(w);
        }

        // accumulate the I/O time,
        time_io_accu += time_io.Elapsed();
        time_io.Reset(); // to be sure we don't count 2x,

        // skip too long utterances (or we run out of memory),
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Utterance too long, skipping! " << utt
            << " (length " << mat.NumRows() << ", max_frames "
            << max_frames << ")";
          num_other_error++;
          continue;
        }

        // correct small length mismatch or drop sentence,
        {
          // add lengths to vector,
          std::vector<int32> length;
          length.push_back(mat.NumRows());
          length.push_back(targets.size());
          length.push_back(weights.Dim());
          // find min, max,
          int32 min = *std::min_element(length.begin(), length.end());
          int32 max = *std::max_element(length.begin(), length.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            // we truncate to shortest,
            if (mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if (targets.size() != min) targets.resize(min);
            if (weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << "Length mismatch! Targets " << targets.size()
                       << ", features " << mat.NumRows() << ", " << utt;
            num_other_error++;
            continue;
          }
        }
        // apply feature transform (if empty, input is copied),
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        // remove frames with '0' weight from training,
        {
          // are there any frames to be removed? (frames with zero weight),
          BaseFloat weight_min = weights.Min();
          KALDI_ASSERT(weight_min >= 0.0);
          if (weight_min == 0.0) {
            // create vector with frame-indices to keep,
            std::vector<MatrixIndexT> keep_frames;
            for (int32 i = 0; i < weights.Dim(); i++) {
              if (weights(i) > 0.0) {
                keep_frames.push_back(i);
              }
            }

            // when all frames are removed, we skip the sentence,
            if (keep_frames.size() == 0) continue;

            // filter feature-frames,
            CuMatrix<BaseFloat> tmp_feats(keep_frames.size(), feats_transf.NumCols());
            tmp_feats.CopyRows(feats_transf, CuArray<MatrixIndexT>(keep_frames));
            tmp_feats.Swap(&feats_transf);

            // filter targets,
            Posterior tmp_targets;
            for (int32 i = 0; i < keep_frames.size(); i++) {
              tmp_targets.push_back(targets[keep_frames[i]]);
            }
            tmp_targets.swap(targets);

            // filter weights,
            Vector<BaseFloat> tmp_weights(keep_frames.size());
            for (int32 i = 0; i < keep_frames.size(); i++) {
              tmp_weights(i) = weights(keep_frames[i]);
            }
            tmp_weights.Swap(&weights);
          }
        }

        // pass data to randomizers,
        KALDI_ASSERT(feats_transf.NumRows() == targets.size());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(targets);
        weights_randomizer.AddData(weights);
        num_done++;

        time_io.Reset(); // reset before reading next feature matrix,
      }

      // randomize,
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask =
          randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches),
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs,
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass,
        nnet.Propagate(nnet_in, &nnet_out);

        // evaluate objective function we've chosen,
        if (objective_function == "xent") {
          // gradients re-scaled by weights in Eval,
          xent.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        } else if (objective_function == "mse") {
          // gradients re-scaled by weights in Eval,
          mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        } else if (0 == objective_function.compare(0, 9, "multitask")) {
          // gradients re-scaled by weights in Eval,
          multitask.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
        }

        if (!crossvalidate) {
          // back-propagate, and do the update,
          nnet.Backpropagate(obj_diff, NULL);
        }

        // 1st mini-batch : show what happens in network,
        if (total_frames == 0) {
          KALDI_LOG << "### After " << total_frames << " frames,";
          KALDI_LOG << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_LOG << nnet.InfoBackPropagate();
            KALDI_LOG << nnet.InfoGradient();
          }
        }

        // VERBOSE LOG
        // monitor the NN training (--verbose=2),
        if (GetVerboseLevel() >= 2) {
          static int32 counter = 0;
          counter += nnet_in.NumRows();
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

        total_frames += nnet_in.NumRows();
      }
    }  // main loop,

    // after last mini-batch : show what happens in network,
    KALDI_LOG << "### After " << total_frames << " frames,";
    KALDI_LOG << nnet.InfoPropagate();
    if (!crossvalidate) {
      KALDI_LOG << nnet.InfoBackPropagate();
      KALDI_LOG << nnet.InfoGradient();
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
      << total_frames / time.Elapsed() << " frames per sec;"
      << " i/o time " << 100.*time_io_accu/time.Elapsed() << "%]";

    if (objective_function == "xent") {
      KALDI_LOG << xent.ReportPerClass();
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
    } else if (0 == objective_function.compare(0, 9, "multitask")) {
      KALDI_LOG << multitask.Report();
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

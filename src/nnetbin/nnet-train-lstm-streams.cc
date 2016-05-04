// nnetbin/nnet-train-lstm-streams.cc

// Copyright 2015-2016  Brno University of Technology (Author: Karel Vesely)
//           2014  Jiayu DU (Jerry), Wei Li

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

#include <numeric>

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
        "Perform one iteration of LSTM training by Stochastic Gradient Descent.\n"
        "The training targets are pdf-posteriors, usually prepared by ali-to-post.\n"
        "The updates are per-utterance.\n"
        "\n"
        "Usage: nnet-train-lstm-streams [options] "
          "<feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: nnet-train-lstm-streams scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
        "Perform cross-validation (don't back-propagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in Nnet format");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,
        "Objective function : xent|mse");

    /*
    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
      "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
      "Per-frame weights to scale gradients (frame selection/weighting).");
    */

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    // <jiayu>
    int32 targets_delay = 5;
    po.Register("targets-delay", &targets_delay, "[LSTM] BPTT targets delay");

    int32 batch_size = 20;
    po.Register("batch-size", &batch_size, "[LSTM] BPTT batch size");

    int32 num_stream = 4;
    po.Register("num-stream", &num_stream, "[LSTM] BPTT multi-stream training");
    // </jiayu>

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

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader target_reader(targets_rspecifier);

    /*
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    */

    Xent xent;
    Mse mse;

    Timer time;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    int32 num_done = 0,
          num_no_tgt_mat = 0,
          num_other_error = 0;

    // book-keeping for multi-streams,
    std::vector<std::string> keys(num_stream);
    std::vector<Matrix<BaseFloat> > feats(num_stream);
    std::vector<Posterior> targets(num_stream);
    std::vector<int32> curt(num_stream, 0);
    std::vector<int32> lent(num_stream, 0);
    std::vector<int32> new_utt_flags(num_stream, 0);

    // bptt batch buffer,
    int32 feat_dim = nnet.InputDim();
    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
    Matrix<BaseFloat> feat(batch_size * num_stream, feat_dim, kSetZero);
    Posterior target(batch_size * num_stream);
    CuMatrix<BaseFloat> feat_transf, nnet_out, obj_diff;

    while (1) {
      // loop over all streams, check if any stream reaches the end of its utterance,
      // if any, feed the exhausted stream with a new utterance, update book-keeping infos
      for (int s = 0; s < num_stream; s++) {
        // this stream still has valid frames
        if (curt[s] < lent[s]) {
          new_utt_flags[s] = 0;
          continue;
        }
        // else, this stream exhausted, need new utterance
        while (!feature_reader.Done()) {
          const std::string& key = feature_reader.Key();
          // get the feature matrix,
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          // forward the features through a feature-transform,
          nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feat_transf);

          // get the labels,
          if (!target_reader.HasKey(key)) {
            KALDI_WARN << key << ", missing targets";
            num_no_tgt_mat++;
            feature_reader.Next();
            continue;
          }
          const Posterior& target = target_reader.Value(key);

          // check that the length matches,
          if (feat_transf.NumRows() != target.size()) {
            KALDI_WARN << key
              << ", length miss-match between feats and targets, skipping";
            num_other_error++;
            feature_reader.Next();
            continue;
          }

          // checks ok, put the data in the buffers,
          keys[s] = key;
          feats[s].Resize(feat_transf.NumRows(), feat_transf.NumCols());
          feat_transf.CopyToMat(&feats[s]);
          targets[s] = target;
          curt[s] = 0;
          lent[s] = feats[s].NumRows();
          new_utt_flags[s] = 1;  // a new utterance feeded to this stream
          feature_reader.Next();
          break;
        }
      }

      // we are done if all streams are exhausted
      int done = 1;
      for (int s = 0; s < num_stream; s++) {
        // this stream still contains valid data, not yet exhausted,
        if (curt[s] < lent[s]) done = 0;
      }
      if (done) break;

      // fill a multi-stream bptt batch
      // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
      // * target: padded to batch_size
      // * feat: first shifted to achieve targets delay; then padded to batch_size
      for (int t = 0; t < batch_size; t++) {
        for (int s = 0; s < num_stream; s++) {
          // frame_mask & targets padding
          if (curt[s] < lent[s]) {
            frame_mask(t * num_stream + s) = 1;
            target[t * num_stream + s] = targets[s][curt[s]];
          } else {
            frame_mask(t * num_stream + s) = 0;
            target[t * num_stream + s] = targets[s][lent[s]-1];
          }
          // feat shifting & padding
          if (curt[s] + targets_delay < lent[s]) {
            feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s] + targets_delay));
          } else {
            feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(lent[s] - 1));
          }
          curt[s]++;
        }
      }

      // for streams with new utterance, history states need to be reset
      nnet.ResetLstmStreams(new_utt_flags);

      // forward pass
      nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);

      // evaluate objective function we've chosen,
      if (objective_function == "xent") {
        xent.Eval(frame_mask, nnet_out, target, &obj_diff);
      } else if (objective_function == "mse") {
        mse.Eval(frame_mask, nnet_out, target, &obj_diff);
      } else {
        KALDI_ERR << "Unknown objective function code : "
                  << objective_function;
      }

      if (!crossvalidate) {
        // back-propagate, and do the update,
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
      if (kaldi::g_kaldi_verbose_level >= 2) {
        static int32 counter = 0;
        counter += frame_mask.Sum();
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

      num_done +=
        std::accumulate(new_utt_flags.begin(), new_utt_flags.end(), 0);

      total_frames += frame_mask.Sum();

      {  // do this every 5000 uttearnces,
        static int32 utt_counter = 0;
        utt_counter +=
          std::accumulate(new_utt_flags.begin(), new_utt_flags.end(), 0);
        if (utt_counter > 5000) {
          utt_counter = 0;
          // report speed,
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: "
            << "time elapsed = " << time_now / 60 << " min; "
            << "processed " << total_frames / time_now << " frames per sec.";
#if HAVE_CUDA == 1
          // check that GPU computes accurately,
          CuDevice::Instantiate().CheckGpuHealth();
#endif
        }
      }
    }

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

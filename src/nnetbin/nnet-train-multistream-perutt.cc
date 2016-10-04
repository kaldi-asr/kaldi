// nnetbin/nnet-train-multistream-perutt.cc

// Copyright 2016 Brno University of Technology (author: Karel Vesely)
// Copyright 2015 Chongjia Ni

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
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
#include "nnet/nnet-matrix-buffer.h"

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

#include <numeric>
#include <algorithm>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
      "Perform one iteration of Multi-stream training, per-utterance BPTT for (B)LSTMs.\n"
      "The updates are done per-utterance, while several utterances are \n"
      "processed at the same time.\n"
      "\n"
      "Usage: nnet-train-multistream-perutt [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
      "e.g.: nnet-train-blstm-streams scp:feats.scp ark:targets.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    // training options,
    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write model in binary mode");

    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
        "Perform cross-validation (no backpropagation)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in Nnet format");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
        "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
        "Per-frame weights to scale gradients (frame selection/weighting).");

    int32 num_streams = 20;
    po.Register("num-streams", &num_streams,
        "Number of sentences processed in parallel (can be lower if sentences are long)");

    double max_frames = 8000;
    po.Register("max-frames", &max_frames,
        "Max number of frames to be processed");

    bool dummy = false;
    po.Register("randomize", &dummy, "Dummy option.");

    std::string use_gpu = "yes";
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
    if ( feature_transform != "" ) {
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

    // Initialize feature and target readers,
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }


    Xent xent;

    CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    // Buffer for input features, used for choosing utt's with similar length,
    MatrixBuffer matrix_buffer;
    matrix_buffer.Init(&feature_reader);

    int32 num_done = 0,
          num_no_tgt_mat = 0,
          num_other_error = 0;

    while (!matrix_buffer.Done()) {

      // Fill the parallel data into 'std::vector',
      std::vector<Matrix<BaseFloat> > feats_utt;
      std::vector<Posterior> labels_utt;
      std::vector<Vector<BaseFloat> > weights_utt;
      std::vector<int32> frame_num_utt;
      {
        matrix_buffer.ResetLength();  ///< reset the 'preferred' length,
        for (matrix_buffer.Next(); !matrix_buffer.Done(); matrix_buffer.Next()) {
          std::string utt = matrix_buffer.Key();
          // Check that we have targets,
          if (!targets_reader.HasKey(utt)) {
            KALDI_WARN << utt << ", missing targets";
            num_no_tgt_mat++;
            continue;
          }
          // Do we have frame-weights?
          if (frame_weights != "" && !weights_reader.HasKey(utt)) {
            KALDI_WARN << utt << ", missing frame-weights";
            num_other_error++;
            continue;
          }

          // Get feature / target pair,
          Matrix<BaseFloat> mat = matrix_buffer.Value();
          Posterior targets  = targets_reader.Value(utt);

          // Skip too long sentences,
          if (mat.NumRows() > max_frames) continue;

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
              KALDI_WARN << "Length mismatch! Targets " << targets.size()
                         << ", features " << mat.NumRows() << ", " << utt;
              num_other_error++;
              continue;
            }
          }

          // input transform may contain splicing,
          nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

          // store,
          feats_utt.push_back(Matrix<BaseFloat>(feats_transf));
          labels_utt.push_back(targets);
          weights_utt.push_back(weights);
          frame_num_utt.push_back(feats_transf.NumRows());

          if (frame_num_utt.size() == num_streams) break;

          // See how many frames we'd have (after padding), if we add one more utterance,
          int32 max = (*std::max_element(frame_num_utt.begin(), frame_num_utt.end()));
          if (max * (frame_num_utt.size() + 1) > max_frames) break;
        }
      }
      // Having no data? Skip the cycle...
      if (frame_num_utt.size() == 0) continue;

      // Pack the parallel data,
      Matrix<BaseFloat> feat_mat_host;
      Posterior target_host;
      Vector<BaseFloat> weight_host;
      {
        // Number of sequences,
        int32 n_streams = frame_num_utt.size();
        int32 frame_num_padded = (*std::max_element(frame_num_utt.begin(), frame_num_utt.end()));
        int32 feat_dim = feats_utt.front().NumCols();

        // Create the final feature matrix. Every utterance is padded to the max
        // length within this group of utterances,
        feat_mat_host.Resize(n_streams * frame_num_padded, feat_dim, kSetZero);
        target_host.resize(n_streams * frame_num_padded);
        weight_host.Resize(n_streams * frame_num_padded, kSetZero);

        for (int32 s = 0; s < n_streams; s++) {
          const Matrix<BaseFloat>& mat_tmp = feats_utt[s];
          for (int32 r = 0; r < frame_num_utt[s]; r++) {
            feat_mat_host.Row(r*n_streams + s).CopyFromVec(mat_tmp.Row(r));
          }
        }

        for (int32 s = 0; s < n_streams; s++) {
          const Posterior& target_tmp = labels_utt[s];
          for (int32 r = 0; r < frame_num_utt[s]; r++) {
            target_host[r*n_streams + s] = target_tmp[r];
          }
        }

        // padded frames will keep initial zero-weight,
        for (int32 s = 0; s < n_streams; s++) {
          const Vector<BaseFloat>& weight_tmp = weights_utt[s];
          for (int32 r = 0; r < frame_num_utt[s]; r++) {
            weight_host(r*n_streams + s) = weight_tmp(r);
          }
        }
      }

      // Set the original lengths of utterances before padding,
      nnet.SetSeqLengths(frame_num_utt);
      // Show the 'utt' lengths in the VLOG[2],
      if (GetVerboseLevel() >= 2) {
        KALDI_LOG << "frame_num_utt[" << frame_num_utt.size() << "]" << frame_num_utt;
      }
      // Reset all the streams (we have new sentences),
      nnet.ResetStreams(std::vector<int32>(frame_num_utt.size(), 1));

      // Propagation,
      nnet.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &nnet_out);

      // Per-frame cross-entropy, gradients get re-scaled by weights,
      xent.Eval(weight_host, nnet_out, target_host, &obj_diff);

      // Backward pass
      if (!crossvalidate) {
        nnet.Backpropagate(obj_diff, NULL);
      }

      // 1st model update : show what happens in network,
      if (total_frames == 0) {
        KALDI_VLOG(1) << "### After " << total_frames << " frames,";
        KALDI_VLOG(1) << nnet.Info();
        KALDI_VLOG(1) << nnet.InfoPropagate();
        if (!crossvalidate) {
          KALDI_VLOG(1) << nnet.InfoBackPropagate();
          KALDI_VLOG(1) << nnet.InfoGradient();
        }
      }

      int32 tmp_done = num_done;
      kaldi::int64 tmp_frames = total_frames;

      num_done += frame_num_utt.size();
      total_frames += std::accumulate(frame_num_utt.begin(), frame_num_utt.end(), 0);

      // report the speed,
      int32 N = 5000;
      if (tmp_done / N != num_done / N) {
        double time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances, "
          << "(" << total_frames/360000.0 << "h), "
          << "time elapsed = " << time_now / 60 << " min; "
          << "processed " << total_frames / time_now << " frames per sec.";
      }

      // monitor the NN training (--verbose=2),
      int32 F = 25000;
      if (GetVerboseLevel() >= 3) {
        // print every 25k frames,
        if (tmp_frames / F != total_frames / F) {
          KALDI_VLOG(3) << "### After " << total_frames << " frames,";
          KALDI_VLOG(3) << nnet.Info();
          KALDI_VLOG(3) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(3) << nnet.InfoBackPropagate();
            KALDI_VLOG(3) << nnet.InfoGradient();
          }
        }
      }
    }

    // after last model update : show what happens in network,
    if (GetVerboseLevel() >= 1) {  // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.Info();
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << xent.ReportPerClass();
    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << ", " << time.Elapsed() / 60 << " min, "
              << "fps" << total_frames / time.Elapsed() << "]";
    KALDI_LOG << xent.Report();

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

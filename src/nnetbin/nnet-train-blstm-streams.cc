// nnetbin/nnet-train-blstm-streams.cc

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
        "Perform one iteration of senones training by SGD.\n"
        "The updates are done per-utternace and by processing multiple utterances in parallel.\n"
        "\n"
        "Usage: nnet-train-blstm-streams [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-blstm-streams scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);
    // training options
    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true,
         crossvalidate = false;
    po.Register("binary", &binary, "Write model  in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (no backpropagation)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 num_streams = 4;
    po.Register("num_streams", &num_streams, "Number of sequences processed in parallel");

    double frame_limit = 100000;
    po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");

    int32 report_step = 100;
    po.Register("report-step", &report_step, "Step (number of sequences) for status reporting");

    std::string use_gpu = "yes";
    // po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
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
    Vector<BaseFloat> weights;
    // Select the GPU
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

    kaldi::int64 total_frames = 0;

    // Initialize feature ans labels readers
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
    // Feature matrix of every utterance
    std::vector< Matrix<BaseFloat> > feats_utt(num_streams);
    // Label vector of every utterance
    std::vector< Posterior > labels_utt(num_streams);
    std::vector< Vector<BaseFloat> > weights_utt(num_streams);

    int32 feat_dim = nnet.InputDim();

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (1) {

      std::vector<int32> frame_num_utt;
      int32 sequence_index = 0, max_frame_num = 0;

      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        // Check that we have targets
        if (!targets_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
        // Get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior targets  = targets_reader.Value(utt);

        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else {  // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }
        // correct small length mismatch ... or drop sentence
        {
          // add lengths to vector
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back(targets.size());
          lenght.push_back(weights.Dim());
          // find min, max
          int32 min = *std::min_element(lenght.begin(), lenght.end());
          int32 max = *std::max_element(lenght.begin(), lenght.end());
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

        if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();
        feats_utt[sequence_index] = mat;
        labels_utt[sequence_index] = targets;
        weights_utt[sequence_index] = weights;

        frame_num_utt.push_back(mat.NumRows());
        sequence_index++;
        // If the total number of frames reaches frame_limit, then stop adding more sequences, regardless of whether
        // the number of utterances reaches num_sequence or not.
        if (frame_num_utt.size() == num_streams || frame_num_utt.size() * max_frame_num > frame_limit) {
            feature_reader.Next(); break;
        }
      }
      int32 cur_sequence_num = frame_num_utt.size();

      // Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
      Matrix<BaseFloat> feat_mat_host(cur_sequence_num * max_frame_num, feat_dim, kSetZero);
      Posterior target_host;
      Vector<BaseFloat> weight_host;

      target_host.resize(cur_sequence_num * max_frame_num);
      weight_host.Resize(cur_sequence_num * max_frame_num, kSetZero);

      for (int s = 0; s < cur_sequence_num; s++) {
        Matrix<BaseFloat> mat_tmp = feats_utt[s];
        for (int r = 0; r < frame_num_utt[s]; r++) {
          feat_mat_host.Row(r*cur_sequence_num + s).CopyFromVec(mat_tmp.Row(r));
        }
      }

      for (int s = 0; s < cur_sequence_num; s++) {
        Posterior target_tmp = labels_utt[s];
        for (int r = 0; r < frame_num_utt[s]; r++) {
          target_host[r*cur_sequence_num+s] = target_tmp[r];
        }
        Vector<BaseFloat> weight_tmp = weights_utt[s];
        for (int r = 0; r < frame_num_utt[s]; r++) {
          weight_host(r*cur_sequence_num+s) = weight_tmp(r);
        }
      }

      // transform feature
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat_mat_host), &feats_transf);

      // Set the original lengths of utterances before padding
      nnet.SetSeqLengths(frame_num_utt);

      // Propagation and xent training
      nnet.Propagate(feats_transf, &nnet_out);

      if (objective_function == "xent") {
          // gradients re-scaled by weights in Eval,
          xent.Eval(weight_host, nnet_out, target_host, &obj_diff);
      } else if (objective_function == "mse") {
          // gradients re-scaled by weights in Eval,
          mse.Eval(weight_host, nnet_out, target_host, &obj_diff);
      } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
      }

      // Backward pass
      if (!crossvalidate) {
        nnet.Backpropagate(obj_diff, NULL);
      }

      // 1st minibatch : show what happens in network
      if (kaldi::g_kaldi_verbose_level >= 2 && total_frames == 0) {  // vlog-1
        KALDI_VLOG(1) << "### After " << total_frames << " frames,";
        KALDI_VLOG(1) << nnet.InfoPropagate();
        if (!crossvalidate) {
          KALDI_VLOG(1) << nnet.InfoBackPropagate();
          KALDI_VLOG(1) << nnet.InfoGradient();
        }
      }

      num_done += cur_sequence_num;
      total_frames += feats_transf.NumRows();

      if (feature_reader.Done()) break;  // end loop of while(1)
    }

    // Check network parameters and gradients when training finishes
    if (kaldi::g_kaldi_verbose_level >= 1) {  // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";
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

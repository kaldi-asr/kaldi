// nnetbin/nnet-train-multistream.cc

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


namespace kaldi {

bool ReadData(SequentialBaseFloatMatrixReader& feature_reader,
              RandomAccessPosteriorReader& target_reader,
              RandomAccessBaseFloatVectorReader& weights_reader,
              int32 length_tolerance,
              Matrix<BaseFloat>* feats,
              Posterior* targets,
              Vector<BaseFloat>* weights,
              int32* num_no_tgt_mat,
              int32* num_other_error) {

  // We're looking for the 1st valid utterance...
  for ( ; !feature_reader.Done(); feature_reader.Next()) {
    // Do we have targets?
    const std::string& utt = feature_reader.Key();
    if (!target_reader.HasKey(utt)) {
      KALDI_WARN << utt << ", missing targets";
      (*num_no_tgt_mat)++;
      continue;
    }
    // Do we have frame-weights?
    if (weights_reader.IsOpen() && !weights_reader.HasKey(utt)) {
      KALDI_WARN << utt << ", missing frame-weights";
      (*num_other_error)++;
      continue;
    }

    // get the (feature,target) pair,
    (*feats) = feature_reader.Value();
    (*targets) = target_reader.Value(utt);

    // getting per-frame weights,
    if (weights_reader.IsOpen()) {
      (*weights) = weights_reader.Value(utt);
    } else {  // all per-frame weights are 1.0
      weights->Resize(feats->NumRows());
      weights->Set(1.0);
    }

    // correct small length mismatch ... or drop sentence
    {
      // add lengths to vector
      std::vector<int32> length;
      length.push_back(feats->NumRows());
      length.push_back(targets->size());
      length.push_back(weights->Dim());
      // find min, max
      int32 min = *std::min_element(length.begin(), length.end());
      int32 max = *std::max_element(length.begin(), length.end());
      // fix or drop ?
      if (max - min < length_tolerance) {
        if (feats->NumRows() != min) feats->Resize(min, feats->NumCols(), kCopyData);
        if (targets->size() != min) targets->resize(min);
        if (weights->Dim() != min) weights->Resize(min, kCopyData);
      } else {
        KALDI_WARN << "Length mismatch! Targets " << targets->size()
                   << ", features " << feats->NumRows() << ", " << utt;
        num_other_error++;
        continue;
      }
    }

    // By getting here we got a valid utterance,
    feature_reader.Next();
    return true;
  }

  // No more data,
  return false;
}

}  // namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform one iteration of Multi-stream training, truncated BPTT for LSTMs.\n"
        "The training targets are pdf-posteriors, usually prepared by ali-to-post.\n"
        "The updates are per-utterance.\n"
        "\n"
        "Usage: nnet-train-multistream [options] "
          "<feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: nnet-train-lstm-streams scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

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

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
      "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
      "Per-frame weights to scale gradients (frame selection/weighting).");

    int32 batch_size = 20;
    po.Register("batch-size", &batch_size,
      "Length of 'one stream' in the Multi-stream training");

    int32 num_streams = 4;
    po.Register("num-streams", &num_streams,
      "Number of streams in the Multi-stream training");

    bool dummy = false;
    po.Register("randomize", &dummy, "Dummy option.");

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
    RandomAccessPosteriorReader target_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    Xent xent(loss_opts);
    Mse mse(loss_opts);

    Timer time;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    int32 num_done = 0,
          num_no_tgt_mat = 0,
          num_other_error = 0;

    // book-keeping for multi-stream training,
    std::vector<Matrix<BaseFloat> > feats_utt(num_streams);
    std::vector<Posterior> labels_utt(num_streams);
    std::vector<Vector<BaseFloat> > weights_utt(num_streams);
    std::vector<int32> new_utt_flags(num_streams);

    CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;

    // MAIN LOOP,
    while (1) {

      // Re-fill the streams, if needed,
      new_utt_flags.assign(num_streams, 0);  // set new-utterance flags to zero,
      for (int s = 0; s < num_streams; s++) {
        // Need a new utterance for stream 's'?
        if (feats_utt[s].NumRows() == 0) {
          Matrix<BaseFloat> feats;
          Posterior targets;
          Vector<BaseFloat> weights;
          // get the data from readers,
          if (ReadData(feature_reader, target_reader, weights_reader,
                       length_tolerance,
                       &feats, &targets, &weights,
                       &num_no_tgt_mat, &num_other_error)) {

            // input transform may contain splicing,
            nnet_transf.Feedforward(CuMatrix<BaseFloat>(feats), &feats_transf);

            /* Here we could do the 'targets_delay', BUT...
             * It is better to do it by a <Splice> component!
             *
             * The prototype would look like this (6th frame becomes 1st frame, etc.):
             * '<Splice> <InputDim> dim1 <OutputDim> dim1 <BuildVector> 5 </BuildVector>'
             */

            // store,
            feats_utt[s] = Matrix<BaseFloat>(feats_transf);
            labels_utt[s] = targets;
            weights_utt[s] = weights;
            new_utt_flags[s] = 1;
          }
        }
      }

      // end the training after processing all the frames,
      size_t frames_to_go = 0;
      for (int32 s = 0; s < num_streams; s++) {
        frames_to_go += feats_utt[s].NumRows();
      }
      if (frames_to_go == 0) break;

      // number of frames we'll pack as the streams,
      std::vector<int32> frame_num_utt;

      // pack the parallel data,
      Matrix<BaseFloat> feat_mat_host;
      Posterior target_host;
      Vector<BaseFloat> weight_host;
      {
        // Number of sequences (can have zero length),
        int32 n_streams = num_streams;

        // Create the final feature matrix with 'interleaved feature-lines',
        feat_mat_host.Resize(n_streams * batch_size, nnet.InputDim(), kSetZero);
        target_host.resize(n_streams * batch_size);
        weight_host.Resize(n_streams * batch_size, kSetZero);
        frame_num_utt.resize(n_streams, 0);

        // we'll slice at most 'batch_size' frames,
        for (int32 s = 0; s < n_streams; s++) {
          int32 num_rows = feats_utt[s].NumRows();
          frame_num_utt[s] = std::min(batch_size, num_rows);
        }

        // pack the data,
        {
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

        // remove the data we just packed,
        {
          for (int32 s = 0; s < n_streams; s++) {
            // feats,
            Matrix<BaseFloat>& m = feats_utt[s];
            if (m.NumRows() == frame_num_utt[s]) {
              feats_utt[s].Resize(0,0);  // we packed last chunk,
            } else {
              feats_utt[s] = Matrix<BaseFloat>(
                m.RowRange(frame_num_utt[s], m.NumRows() - frame_num_utt[s])
              );
            }
            // labels,
            Posterior& post = labels_utt[s];
            post.erase(post.begin(), post.begin() + frame_num_utt[s]);
            // weights,
            Vector<BaseFloat>& w = weights_utt[s];
            if (w.Dim() == frame_num_utt[s]) {
              weights_utt[s].Resize(0);  // we packed last chunk,
            } else {
              weights_utt[s] = Vector<BaseFloat>(
                w.Range(frame_num_utt[s], w.Dim() - frame_num_utt[s])
              );
            }
          }
        }
      }

      // pass the info about padding,
      nnet.SetSeqLengths(frame_num_utt);
      // Show the 'utt' lengths in the VLOG[2],
      if (GetVerboseLevel() >= 2) {
        std::ostringstream os;
        os << "[ ";
        for (size_t i = 0; i < frame_num_utt.size(); i++) {
          os << frame_num_utt[i] << " ";
        }
        os << "]";
        KALDI_LOG << "frame_num_utt[" << frame_num_utt.size() << "]" << os.str();
      }

      // with new utterance we reset the history,
      nnet.ResetStreams(new_utt_flags);

      // forward pass,
      nnet.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &nnet_out);

      // evaluate objective function we've chosen,
      if (objective_function == "xent") {
        xent.Eval(weight_host, nnet_out, target_host, &obj_diff);
      } else if (objective_function == "mse") {
        mse.Eval(weight_host, nnet_out, target_host, &obj_diff);
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
        KALDI_LOG << "### After " << total_frames << " frames,";
        KALDI_LOG << nnet.Info();
        KALDI_LOG << nnet.InfoPropagate();
        if (!crossvalidate) {
          KALDI_LOG << nnet.InfoBackPropagate();
          KALDI_LOG << nnet.InfoGradient();
        }
      }

      kaldi::int64 tmp_frames = total_frames;

      num_done += std::accumulate(new_utt_flags.begin(), new_utt_flags.end(), 0);
      total_frames += std::accumulate(frame_num_utt.begin(), frame_num_utt.end(), 0);

      // monitor the NN training (--verbose=2),
      int32 F = 25000;
      if (GetVerboseLevel() >= 2) {
        // print every 25k frames,
        if (tmp_frames / F != total_frames / F) {
          KALDI_VLOG(2) << "### After " << total_frames << " frames,";
          KALDI_VLOG(2) << nnet.Info();
          KALDI_VLOG(2) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(2) << nnet.InfoBackPropagate();
            KALDI_VLOG(2) << nnet.InfoGradient();
          }
        }
      }
    }

    // after last minibatch : show what happens in network,
    KALDI_LOG << "### After " << total_frames << " frames,";
    KALDI_LOG << nnet.Info();
    KALDI_LOG << nnet.InfoPropagate();
    if (!crossvalidate) {
      KALDI_LOG << nnet.InfoBackPropagate();
      KALDI_LOG << nnet.InfoGradient();
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    if (objective_function == "xent") {
      KALDI_LOG << xent.ReportPerClass();
    }

    KALDI_LOG << "Done " << num_done << " files, "
      << num_no_tgt_mat << " with no tgt_mats, "
      << num_other_error << " with other errors. "
      << "[" << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
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

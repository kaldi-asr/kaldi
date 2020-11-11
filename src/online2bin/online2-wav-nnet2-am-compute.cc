// online2bin/online2-wav-nnet2-am-compute.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2014  David Snyder

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

#include "feat/wave-reader.h"
#include "online2/online-nnet2-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Simulates the online neural net computation for each file of input\n"
        "features, and outputs as a matrix the result, with optional\n"
        "iVector-based speaker adaptation. Note: some configuration values\n"
        "and inputs are set via config files whose filenames are passed as\n"
        "options.  Used mostly for debugging.\n"
        "Note: if you want it to apply a log (e.g. for log-likelihoods), use\n"
        "--apply-log=true.\n"
        "\n"
        "Usage:  online2-wav-nnet2-am-compute [options] <nnet-in>\n"
        "<spk2utt-rspecifier> <wav-rspecifier> <feature-or-loglikes-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to compute utterance by utterance.\n";

    BaseFloat chunk_length_secs = 0.05;
    bool apply_log = false;
    bool pad_input = true;
    bool online = true;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    ParseOptions po(usage);
    po.Register("apply-log", &apply_log, "Apply a log to the result of the computation "
                "before outputting.");
    po.Register("pad-input", &pad_input, "If true, duplicate the first and last frames "
                "of input features as required for temporal context, to prevent #frames "
                "of output being less than those of input.");
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");

    feature_opts.Register(&po);
    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet2_rxfilename = po.GetArg(1),
        spk2utt_rspecifier = po.GetArg(2),
        wav_rspecifier = po.GetArg(3),
        features_or_loglikes_wspecifier = po.GetArg(4);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_opts.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_opts.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);

    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet2_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }
    Nnet &nnet = am_nnet.GetNnet();

    int64 num_done = 0, num_frames = 0;
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatCuMatrixWriter writer(features_or_loglikes_wspecifier);

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
      OnlineCmvnState cmvn_state(global_cmvn_stats);

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        feature_pipeline.SetCmvnState(cmvn_state);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }
        }

        int32 feats_num_frames = feature_pipeline.NumFramesReady(),
              feats_dim = feature_pipeline.Dim();
        Matrix<BaseFloat> feats(feats_num_frames, feats_dim);

        for (int32 i = 0; i < feats_num_frames; i++) {
          SubVector<BaseFloat> frame_vector(feats, i);
          feature_pipeline.GetFrame(i, &frame_vector);
        }

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);
        feature_pipeline.GetCmvnState(&cmvn_state);

        int32 output_frames = feats.NumRows(),
              output_dim = nnet.OutputDim();
        CuMatrix<BaseFloat> output(output_frames, output_dim),
                            feats_cu(feats);

        if (!pad_input)
          output_frames -= nnet.LeftContext() + nnet.RightContext();
        if (output_frames <= 0) {
          KALDI_WARN << "Skipping utterance " << utt << " because output "
                     << "would be empty.";
          continue;
        }

        NnetComputation(nnet, feats_cu, pad_input, &output);

        if (apply_log) {
          output.ApplyFloor(1.0e-20);
          output.ApplyLog();
        }

        writer.Write(utt, output);
        num_frames += feats.NumRows();
        num_done++;

        KALDI_LOG << "Processed data for utterance " << utt;
      }
    }

    KALDI_LOG << "Processed " << num_done << " feature files, "
              << num_frames << " frames of input were processed.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
} // main()

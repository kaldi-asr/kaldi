// online2/compute-online-feats.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "feat/wave-reader.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract features and ivectors for utterances using the online feature "
        "extractor but in an offline mode\n"
        "\n"
        "Usage:  compute-online-feats [options] <wave-rspecifier> "
        "<ivector-wspecifier> <feats-wspecifier>\n"
        "e.g.: \n"
        "  ./compute-online-feats --config=feature_config wav.scp "
        "ark,scp:ivector.ark,ivector.scp ark,scp:feat.ark,feat.scp\n";

    ParseOptions po(usage);
    // Use online feature config as that is the flow we are trying to model
    OnlineNnet2FeaturePipelineConfig feature_opts;

    feature_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
                ivector_wspecifier = po.GetArg(2),
                feature_wspecifier = po.GetArg(3);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatVectorWriter ivector_writer;
    BaseFloatMatrixWriter feature_writer;

    if (!ivector_writer.Open(ivector_wspecifier)) {
      KALDI_ERR << "Could not initialize ivector_writer with wspecifier "
                << ivector_wspecifier;
    }
    if (!feature_writer.Open(feature_wspecifier)) {
      KALDI_ERR << "Could not initialize feature_writer with wspecifier "
                << feature_wspecifier;
    }

    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      KALDI_LOG << "Processing Utterance " << utt;
      try {
        OnlineNnet2FeaturePipeline feature_extractor(feature_info);

        const WaveData &wave_data = reader.Value();
        SubVector<BaseFloat> waveform(wave_data.Data(), 0);

        feature_extractor.AcceptWaveform(wave_data.SampFreq(), waveform);
        feature_extractor.InputFinished();

        int numFrames = feature_extractor.NumFramesReady();
        int32 feat_dim = feature_extractor.InputFeature()->Dim();

        // create list of frames
        std::vector<int> frames(numFrames);
        for (int j = 0; j < numFrames; j++) frames[j] = j;

        Matrix<BaseFloat> features(numFrames, feat_dim);

        feature_extractor.InputFeature()->GetFrames(frames, &features);
        feature_writer.Write(utt, features);

        Vector<BaseFloat> ivector;

        if (feature_extractor.IvectorFeature() != NULL) {
          int32 ivector_dim = feature_extractor.IvectorFeature()->Dim();
          ivector.Resize(ivector_dim);
          feature_extractor.IvectorFeature()->GetFrame(numFrames - 1, &ivector);
          ivector_writer.Write(utt, ivector);
        }

        num_success++;
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance " << utt;
        continue;
      }
    }
    KALDI_LOG << "Processed " << num_utts << " utterances with "
              << num_utts - num_success << " failures.";
    return (num_success != 0 ? 0 : 1);

  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

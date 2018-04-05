// online2bin/online2-wav-dump-features.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    
    const char *usage =
        "Reads in wav file(s) and processes them as in online2-wav-nnet2-latgen-faster,\n"
        "but instead of decoding, dumps the features.  Most of the parameters\n"
        "are set via configuration variables.\n"
        "\n"
        "Usage: online2-wav-dump-features [options] <spk2utt-rspecifier> <wav-rspecifier> <feature-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to generate features utterance by utterance.\n"
        "Alternate usage: online2-wav-dump-features [options] --print-ivector-dim=true\n"
        "See steps/online/nnet2/{dump_nnet_activations,get_egs.sh} for examples.\n";
    
    ParseOptions po(usage);
    
    // feature_config includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_config;  
    BaseFloat chunk_length_secs = 0.05;
    bool print_ivector_dim = false;
    
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("print-ivector-dim", &print_ivector_dim,
                "If true, print iVector dimension (possibly zero) and exit.  This "
                "version requires no arguments.");
    
    feature_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (!print_ivector_dim && po.NumArgs() != 3) {
      po.PrintUsage();
      return 1;
    }

    OnlineNnet2FeaturePipelineInfo feature_info(feature_config);

    if (print_ivector_dim) {
      std::cout << feature_info.IvectorDim() << std::endl;
      exit(0);
    }
    
    std::string spk2utt_rspecifier = po.GetArg(1),
        wav_rspecifier = po.GetArg(2),
        feats_wspecifier = po.GetArg(3);
    
    
    int32 num_done = 0, num_err = 0;
    int64 num_frames_tot = 0;
    
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatMatrixWriter feats_writer(feats_wspecifier);
    
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);
        
        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        std::vector<Vector<BaseFloat> *> feature_data;

        // We retrieve data from the feature pipeline while adding the wav data bit
        // by bit...  for features like pitch features, this may make a
        // difference to what we get, and we want to make sure that the data we
        // get it exactly compatible with online decoding.

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length = int32(samp_freq * chunk_length_secs);
        if (chunk_length == 0) chunk_length = 1;
        
        int32 samp_offset = 0;
        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;
          
          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);
          samp_offset += num_samp;
          if (samp_offset == data.Dim())  // no more input. flush out last frames
            feature_pipeline.InputFinished();
          
          while (static_cast<int32>(feature_data.size()) <
                 feature_pipeline.NumFramesReady()) {
            int32 t = static_cast<int32>(feature_data.size());
            feature_data.push_back(new Vector<BaseFloat>(feature_pipeline.Dim(),
                                                         kUndefined));
            feature_pipeline.GetFrame(t, feature_data.back());
          }
        }
        int32 T = static_cast<int32>(feature_data.size());
        if (T == 0) {
          KALDI_WARN << "Got no frames of data for utterance " << utt;
          num_err++;
          continue;
        }
        Matrix<BaseFloat> feats(T, feature_pipeline.Dim());
        for (int32 t = 0; t < T; t++) {
          feats.Row(t).CopyFromVec(*(feature_data[t]));
          delete feature_data[t];
        }
        num_frames_tot += T;
        feats_writer.Write(utt, feats);
        feature_pipeline.GetAdaptationState(&adaptation_state);
        num_done++;
      }
    }
    KALDI_LOG << "Processed " << num_done << " utterances, "
              << num_err << " with errors; " << num_frames_tot
              << " frames in total.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

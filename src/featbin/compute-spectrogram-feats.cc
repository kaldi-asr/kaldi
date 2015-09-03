// featbin/compute-spectrogram-feats.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-spectrogram.h"
#include "feat/wave-reader.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Create spectrogram feature files.\n"
        "Usage:  compute-spectrogram-feats [options...] <wav-rspecifier> <feats-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    SpectrogramOptions spec_opts;
    bool subtract_mean = false;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    // Define defaults for gobal options
    std::string output_format = "kaldi";

    // Register the option struct
    spec_opts.Register(&po);
    // Register the options
    po.Register("output-format", &output_format, "Format of the output files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each feature file [CMS]; not recommended to do it this way. ");
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments to process (in seconds).");

    // OPTION PARSING ..........................................................
    //

    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);

    std::string output_wspecifier = po.GetArg(2);

    Spectrogram spec(spec_opts);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    TableWriter<HtkMatrixHolder> htk_writer;

    if (output_format == "kaldi") {
      if (!kaldi_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << output_wspecifier;
    } else if (output_format == "htk") {
      if (!htk_writer.Open(output_wspecifier))
        KALDI_ERR << "Could not initialize output with wspecifier "
                  << output_wspecifier;
    } else {
      KALDI_ERR << "Invalid output_format string " << output_format;
    }

    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }

      if (spec_opts.frame_opts.samp_freq != wave_data.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << spec_opts.frame_opts.samp_freq << " but data has "
                  << wave_data.SampFreq() << " (use --sample-frequency "
                  << "option).  Utterance is " << utt;

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        spec.Compute(waveform, &features, NULL);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        continue;
      }
      if (subtract_mean) {
        Vector<BaseFloat> mean(features.NumCols());
        mean.AddRowSumMat(1.0, features);
        mean.Scale(1.0 / features.NumRows());
        for (int32 i = 0; i < features.NumRows(); i++)
          features.Row(i).AddVec(-1.0, mean);
      }
      if (output_format == "kaldi") {
        kaldi_writer.Write(utt, features);
      } else {
        std::pair<Matrix<BaseFloat>, HtkHeader> p;
        p.first.Resize(features.NumRows(), features.NumCols());
        p.first.CopyFromMat(features);
        int32 frame_shift = spec_opts.frame_opts.frame_shift_ms * 10000;
        HtkHeader header = {
          features.NumRows(),
          frame_shift,
          static_cast<int16>(sizeof(float)*features.NumCols()),
          007 | 020000
        };
        p.second = header;
        htk_writer.Write(utt, p);
      }
      if(num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}


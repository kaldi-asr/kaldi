// featbin/wav-to-duration.cc

// Copyright 2013  Daniel Povey

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
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Read wav files and output an archive consisting of a single float:\n"
        "the duration of each one in seconds.\n"
        "Usage:  wav-to-duration [options...] <wav-rspecifier> <duration-wspecifier>\n"
        "E.g.: wav-to-duration scp:wav.scp ark,t:-\n"
        "See also: wav-copy extract-segments feat-to-len\n"
        "Currently this program may output a lot of harmless warnings regarding\n"
        "nonzero exit status of pipes\n";

    bool read_entire_file = false;

    ParseOptions po(usage);

    po.Register("read-entire-file", &read_entire_file, "If true, use regular WaveHolder "
                "instead of WaveInfoHolder to ensure the returned duration is correct.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
        duration_wspecifier = po.GetArg(2);


    double sum_duration = 0.0,
        min_duration = std::numeric_limits<BaseFloat>::infinity(),
        max_duration = 0;
    int32 num_done = 0;

    BaseFloatWriter duration_writer(duration_wspecifier);
    if (read_entire_file) {
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
      for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string key = wav_reader.Key();
        const WaveData &wave_data = wav_reader.Value();
        BaseFloat duration = wave_data.Duration();
        duration_writer.Write(key, duration);

        sum_duration += duration;
        min_duration = std::min<double>(min_duration, duration);
        max_duration = std::max<double>(max_duration, duration);
        num_done++;
      }
    }
    else {
      SequentialTableReader<WaveInfoHolder> wav_reader(wav_rspecifier);
      for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string key = wav_reader.Key();
        const WaveData &wave_data = wav_reader.Value();
        BaseFloat duration = wave_data.Duration();
        duration_writer.Write(key, duration);

        sum_duration += duration;
        min_duration = std::min<double>(min_duration, duration);
        max_duration = std::max<double>(max_duration, duration);
        num_done++;
      }
    }

    KALDI_LOG << "Printed duration for " << num_done << " audio files.";
    if (num_done > 0) {
      KALDI_LOG << "Mean duration was " << (sum_duration / num_done)
                << ", min and max durations were " << min_duration << ", "
                << max_duration;
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


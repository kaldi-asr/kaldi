// featbin/wav-to-volume.cc

// Copyright 2018 AIShell-foundation (Authors: Yong LIU, Jiayu DU)

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

#include <ctime>
#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Read wav files and output the max volume for each wav\n"
        "Usage:  wav-to-volume [options...] <wav-rspecifier> <maxvol-wspecifier>\n"
        "E.g.: wav-to-volume scp:wav.scp ark:-\n"
        "Currently this program may output a lot of harmless warnings regarding\n"
        "nonzero exit status of pipes\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
        maxvol_wspecifier = po.GetArg(2);

    BaseFloat max_vol_all = 0.0;
    int32 num_done = 0;

    BaseFloatWriter maxvol_writer(maxvol_wspecifier);
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string key = wav_reader.Key();
      const WaveData &wave_data = wav_reader.Value();
      const Matrix<BaseFloat> &input_matrix = wave_data.Data();
      BaseFloat samp_freq_input = wave_data.SampFreq();
      int32 num_samp_input = input_matrix.NumCols(),  // #samples in the input
            num_input_channel = input_matrix.NumRows();  // #channels in the input
      // KALDI_ASSERT(num_input_channel == 1);

      Matrix<BaseFloat> copy_matrix(input_matrix);
      copy_matrix.ApplyPowAbs(1);
      BaseFloat max_vol = copy_matrix.Max();
      max_vol_all = max_vol > max_vol_all ? max_vol : max_vol_all;

      maxvol_writer.Write(key, max_vol);
      num_done++;
      KALDI_VLOG(2) << " Printed key= " << key
              << "sampling frequency of input: " << samp_freq_input
              << " #samples: " << num_samp_input
              << " #channel: " << num_input_channel
              << " #max_vol: " << max_vol;
    }

    KALDI_LOG << "Probed volume for " << num_done << " audio files, "
      << " max vol = " << max_vol_all;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

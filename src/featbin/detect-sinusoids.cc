// featbin/detect-sinusoids.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "feat/sinusoid-detection.h"
#include "feat/wave-reader.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Detect sinusoids (one or two at a time) in waveform input and output\n"
        "frame-by-frame information on their frequencies and energies.  Useful\n"
        "as part of DTMF and dialtone detection.  Output is an archive of\n"
        "matrices; for each file, there is a row per frame, containing\n"
        "<signal-energy-per-sample> <frequency1> <energy1> <frequency2> <energy2>\n"
        "where the frequencies and energies may be zero if no sufficiently\n"
        "dominant sinusoid(s) was/were detected.  If two frequencies were\n"
        "detected, frequency1 < frequency2.  See options for more detail on\n"
        "configuration options.\n"
        "\n"
        "Usage: detect-sinusoids [options] <wav-rspecifier> <matrix-wspecifier>\n"
        "e.g.: detect-sinusoids scp:wav.scp ark,t:sinusoids.ark\n";
    
    ParseOptions po(usage);
    MultiSinusoidDetectorConfig config;

    config.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
        matrix_wspecifier = po.GetArg(2);
    
    int32 num_done = 0, num_err = 0;
    
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    
    MultiSinusoidDetector *detector = NULL;
    
    for (; !wav_reader.Done(); wav_reader.Next()) {
      const WaveData &wav_data = wav_reader.Value();
      const Matrix<BaseFloat> &data = wav_data.Data();
      BaseFloat samp_freq = wav_data.SampFreq();
      int32 num_channels = data.NumRows();
      if (num_channels != 1) {
        KALDI_WARN << "detect-sinusoids requires data with one "
                   << "channel. Recording " << wav_reader.Key() << " has "
                   << num_channels << ".  First select one channel of your "
                   << "data (e.g. using sox)";
        num_err++;
        continue;        
      }
      if (samp_freq < config.subsample_freq) {
        KALDI_WARN << "Sampling frequency of data " << wav_reader.Key()
                   << " is too low " << samp_freq << " < "
                   << config.subsample_freq << ".  Reduce --subsample-freq "
                   << "if you want to run on this data.";
        num_err++;
        continue;
      }
          
      if (detector == NULL ||
          samp_freq != detector->SamplingFrequency()) {
        delete detector;
        detector = new MultiSinusoidDetector(config, samp_freq);
      }

      Matrix<BaseFloat> output;
      DetectSinusoids(data.Row(0), detector, &output);

      if (output.NumRows() == 0) {
        KALDI_WARN << "No output for " << wav_reader.Key();
        num_err++;
      } else {
        matrix_writer.Write(wav_reader.Key(), output);
        num_done++;
      }
    }
    delete detector;    
    KALDI_LOG << "Detected sinusoids in " << num_done << " wave files,"
              << num_err << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


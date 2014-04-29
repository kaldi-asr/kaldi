// featbin/compute-kaldi-pitch-feats.cc

// Copyright 2013   Pegah Ghahremani
//                  Johns Hopkins University (author: Daniel Povey)
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
#include "feat/pitch-functions.cc"
#include "feat/wave-reader.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Apply Kaldi pitch extractor, starting from wav input.  Output is 2-dimensional\n"
        "features consisting of (NCCF, pitch in Hz), where NCCF is between -1 and 1, and\n"
        "higher for voiced frames.  You will typically pipe this into\n"
        "process-kaldi-pitch-feats.\n"
        "Usage: compute-kaldi-pitch-feats [options...] <wav-rspecifier> <feats-wspecifier>\n"
        "e.g.\n"
        "compute-kaldi-pitch-feats --sample-frequency=8000 scp:wav.scp ark:- \n";
    
    
    ParseOptions po(usage);
    PitchExtractionOptions pitch_opts;
    int32 channel = -1; // Note: this isn't configurable because it's not a very
                        // good idea to control it this way: better to extract the
                        // on the command line (in the .scp file) using sox or
                        // similar.

    pitch_opts.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string wav_rspecifier = po.GetArg(1),
        feat_wspecifier = po.GetArg(2);

    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string utt = wav_reader.Key();  
      const WaveData &wave_data = wav_reader.Value(); 
      
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {
        KALDI_ASSERT(num_chan > 0); 
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
      
      if (pitch_opts.samp_freq != wave_data.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << pitch_opts.samp_freq << " but data has "
                  << wave_data.SampFreq() << " (use --sample-frequency "
                  << "option).  Utterance is " << utt;
      
      
      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        Compute(pitch_opts, waveform, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute pitch for utterance "
                   << utt;
        continue;
      }

      double tot = features.Sum();
      if (features.NumCols() != 2 || KALDI_ISINF(tot) || KALDI_ISNAN(tot)) {
        KALDI_WARN << "Pitch extraction failed for utterance " << utt
                   << ", num-rows is " << features.NumRows() << ", total is "
                   << tot;
      }
      
      feat_writer.Write(utt, features);
      if (num_done % 50 == 0 && num_done != 0)
        KALDI_VLOG(2) << "Processed " << num_done << " utterances";
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


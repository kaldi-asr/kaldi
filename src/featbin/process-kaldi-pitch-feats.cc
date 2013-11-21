// featbin/process-kaldi-pitch-feats.cc

// Copyright 2013   Pegah Ghahremani
//                  Johns Hopkins University (author:Pegah Ghahremani, Daniel Povey)
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
        "Create Pitch feature files. \n"
        "Usage: process-pitch-feats2 [options...] <wav-rspecifier> <feats-wspecier>\n"
        "This is a rather special-purpose program which processes 2-dimensional\n"
        "features consisting of (prob-of-voicing, pitch) into something suitable\n"
        "Usage:  process-pitch-feats [options...] <feats-rspecifier> <feats-wspecifier>\n";

    
    // construct all the global objects
    ParseOptions po(usage);
    PitchExtractionOptions pitch_opts;
    PostProcessOption postprop_opts;
    int32 channel = -1;
    // Define defaults for gobal options

    // Register the option struct
    pitch_opts.Register(&po);
    postprop_opts.Register(&po); 
    // OPTION PARSING ..........................................................

    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string wav_rspecifier = po.GetArg(1);
    std::string output_wspecifier = po.GetArg(2);

    //Pitch<double> pitch(pitch_opts);
     
    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.

    if (!kaldi_writer.Open(output_wspecifier))
       KALDI_ERR << "Could not initialize output with wspecifier "
                << output_wspecifier;
    
    int32 num_utts = 0, num_success = 0, num_err = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();  
      const WaveData &wave_data = reader.Value(); 
      
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
      
      if (pitch_opts.frame_opts.samp_freq != wave_data.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << pitch_opts.frame_opts.samp_freq << " but data has "
                  << wave_data.SampFreq() << " (use --sample-frequency option)";
      
      
      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        Compute(pitch_opts, waveform, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute pitches for utterance "
                   << utt;
        continue;
      }
      int num_frames = features.NumRows();
      
      Matrix<BaseFloat> processed_feats(features);
      if (postprop_opts.process_pitch) {
        try {
          PostProcessPitch(postprop_opts, features, &processed_feats);
        } catch (...) {
          KALDI_WARN << "Failed to postprocess pitches for utterance "
                     << utt;
          continue;
        }
      }
      
      if (postprop_opts.add_delta_pitch) {
        if (num_frames == 0 && processed_feats.NumCols() != 3) {
          KALDI_WARN << "Feature file has bad size "
                    << processed_feats.NumRows() << " by " << processed_feats.NumCols();
          num_err++;
          continue;
        }
      } else { 
        if (num_frames == 0 && processed_feats.NumCols() != 2) {
          KALDI_WARN << "Feature file has bad size "
                     << processed_feats.NumRows() << " by " << processed_feats.NumCols();
          num_err++;
          continue;
        }
      }
      
      kaldi_writer.Write(utt, processed_feats);
      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << "Done " << num_success << " out of " << num_utts
              << " utterances. ";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


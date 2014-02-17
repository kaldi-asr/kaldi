// dec-wrap/read-wav-test.cc

// Copyright 2013 Ondrej Platek

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
#include <iostream>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Read wav as compute-mfcc-feat.\n"
        "Usage:  read-wav-test [options...] <wav-rspecifier>\n";

    ParseOptions po(usage);
    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(0);
    }

    std::string wav_rspecifier = po.GetArg(1);


    SequentialTableReader<WaveHolder> reader(wav_rspecifier);

    

    int32 num_utts = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      std::cerr << "wave duration " << wave_data.Duration() << std::endl;
      int32 num_chan = wave_data.Data().NumRows(), this_chan = 0;
      std::cerr << "number of channels " << num_chan 
                << "using " << this_chan << " channel " << std::endl;
      std::cerr << "sample frequency " << wave_data.SampFreq() << std::endl;

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      waveform.Write(std::cout, false);
    }
    std::cerr << "Processed" << num_utts << " utterances." << std::endl;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

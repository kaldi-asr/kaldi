// featbin/apply-filters.cc

// Copyright  2016 Pegah Ghahremani

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
#include "feat/wave-reader.h"
#include "feat/signal.h"

namespace kaldi {
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
      "apply filter to wave files supplied via input pip as FIR or IIR filter.\n"
      "If the --inverse=false, it applies filter as FIR filter\n"
      "and if --inverse=true, the inverse of filter applies as IIR filter.\n"
      "Usage: apply-filters [options...] <wav-in-rxfilename> "
      " <spkfilter-rxfilename> <wav-out-wxfilename>\n"
      "e.g. \n"
      "apply-filters --inverse=false --utt2spkfilter=ark:data/train/utt2spkfilter \n"
      " input.wav filter.wav output_1.wav\n";
    ParseOptions po(usage);
    
    bool inverse = false;
    std::string utt2spkfilter_rspecifier = "";
    po.Register("inverse", &inverse,
                "If false, the filter is applied as FIR filter,"
                "otherwise its inverse applied as IIR filter.");
    po.Register("utt2spkfilter", &utt2spkfilter_rspecifier,
                "rspecifier for utterance to spkear-filter list map"
                " used to filter each utterance");
    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string input_wave_file = po.GetArg(2),
      filter_file = po.GetArg(1),
      output_wave_file = po.GetArg(3);

    WaveData input_wave;
    {
      Input ki(input_wave_file);
      input_wave.Read(ki.Stream());
    
    }

    SequentialBaseFloatVectorReader filter_reader(filter_file);
    const Vector<BaseFloat> &lpc_filter = filter_reader.Value();
   
    Vector<BaseFloat> filtered_wav(input_wave.Data().Row(0));
    BaseFloat samp_freq_input = input_wave.SampFreq();
    // If inverse = false, it does FFT-based block Convolution of filter with 
    // long input signal.
    // Otherwise inverse of filter is convolved with input signal.
    // If we use lp coefficients as [1 -a1 -a2 ... ap] as filter
    // convolving input with this filter is like whitening transform.
    // y'[n] = y[n] - sum_{i=1}^p {input_wav[n-i] * lpc_coeffs[i]} 
    //   = conv(y, [1 :-lpc-coeffs])
    FFTbasedBlockConvolveSignals(lpc_filter, &filtered_wav, inverse);
    Matrix<BaseFloat> filtered_wav_mat(1, filtered_wav.Dim());
    filtered_wav_mat.CopyRowsFromVec(filtered_wav);
    WaveData out_wave(samp_freq_input, filtered_wav_mat); 
    Output ko(output_wave_file, false);
    out_wave.Write(ko.Stream());
    return 0; 
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

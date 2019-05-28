// feat/feature-mfcc-test.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek

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


#include <iostream>

#include "feat/feature-mfcc.h"
#include "base/kaldi-math.h"
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"

using namespace kaldi;



static void UnitTestReadWave() {

  std::cout << "=== UnitTestReadWave() ===\n";

  Vector<BaseFloat> v, v2;

  std::cout << "<<<=== Reading waveform\n";

  {
    std::ifstream is("test_data/test.wav", std::ios_base::binary);
    WaveData wave;
    wave.Read(is);
    const Matrix<BaseFloat> data(wave.Data());
    KALDI_ASSERT(data.NumRows() == 1);
    v.Resize(data.NumCols());
    v.CopyFromVec(data.Row(0));
  }

  std::cout << "<<<=== Reading Vector<BaseFloat> waveform, prepared by matlab\n";
  std::ifstream input(
    "test_data/test_matlab.ascii"
  );
  KALDI_ASSERT(input.good());
  v2.Read(input, false);
  v2.Scale(BaseFloat(1.0 / 32768.0));
  input.close();

  std::cout << "<<<=== Comparing freshly read waveform to 'libsndfile' waveform\n";
  KALDI_ASSERT(v.Dim() == v2.Dim());
  for (int32 i = 0; i < v.Dim(); i++) {
    KALDI_ASSERT(v(i) == v2(i));
  }
  std::cout << "<<<=== Comparing done\n";

  // std::cout << "== The Waveform Samples == \n";
  // std::cout << v;

  std::cout << "Test passed :)\n\n";

}



/**
 */
static void UnitTestSimple() {
  std::cout << "=== UnitTestSimple() ===\n";

  Vector<BaseFloat> v(100000);
  Matrix<BaseFloat> m;

  // init with noise
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = (abs( i * 433024253 ) % 65535) - (65535 / 2);
  }

  std::cout << "<<<=== Just make sure it runs... Nothing is compared\n";
  // the parametrization object
  MfccOptions op;
  // trying to have same opts as baseline.
  op.frame_opts.window_type = "rectangular";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 0.0;

  Mfcc mfcc(op);
  // use default parameters

  // compute mfccs.
  mfcc.Compute(v, 1.0, &m);

  // possibly dump
  //   std::cout << "== Output features == \n" << m;
  std::cout << "Test passed :)\n\n";
}


void UnitTestVtln() {
  // Test the function VtlnWarpFreq.
  BaseFloat low_freq = 10, high_freq = 7800,
      vtln_low_cutoff = 20, vtln_high_cutoff = 7400;

  for (size_t i = 0; i < 100; i++) {
    BaseFloat freq = 5000, warp_factor = 0.9 + RandUniform() * 0.2;
    AssertEqual(MelBanks::VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                             low_freq, high_freq, warp_factor,
                             freq),
                freq / warp_factor);

    AssertEqual(MelBanks::VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                             low_freq, high_freq, warp_factor,
                             low_freq),
                low_freq);
    AssertEqual(MelBanks::VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                             low_freq, high_freq, warp_factor,
                             high_freq),
                high_freq);
    BaseFloat freq2 = low_freq + (high_freq-low_freq) * RandUniform(),
        freq3 = freq2 +  (high_freq-freq2) * RandUniform();  // freq3>=freq2
    BaseFloat w2 = MelBanks::VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                                low_freq, high_freq, warp_factor,
                                freq2);
    BaseFloat w3 = MelBanks::VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                                low_freq, high_freq, warp_factor,
                                freq3);
    KALDI_ASSERT(w3 >= w2);  // increasing function.
    BaseFloat w3dash = MelBanks::VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                                    low_freq, high_freq, 1.0,
                                    freq3);
    AssertEqual(w3dash, freq3);
  }
}

static void UnitTestFeat() {
  UnitTestVtln();
  UnitTestReadWave();
  UnitTestSimple();
  std::cout << "Tests succeeded.\n";
}



int main() {
  try {
    for (int i = 0; i < 5; i++)
      UnitTestFeat();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}

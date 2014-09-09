// feat/feature-fbank-test.cc

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

#include "feat/feature-fbank.h"
#include "base/kaldi-math.h"
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"

using namespace kaldi;



static void UnitTestReadWave() {

  std::cout << "=== UnitTestReadWave() ===\n";

  Vector<BaseFloat> v, v2;

  std::cout << "<<<=== Reading waveform\n";

  {
    std::ifstream is("test_data/test.wav");
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
  FbankOptions op;
  // trying to have same opts as baseline.
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "rectangular";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 0.0;
  op.htk_compat = true;
  op.use_energy = true;

  Fbank fbank(op);
  // use default parameters

  // compute fbanks.
  fbank.Compute(v, 1.0, &m, NULL);

  // possibly dump
  //   std::cout << "== Output features == \n" << m;
  std::cout << "Test passed :)\n\n";
}


static void UnitTestHTKCompare1() {
  std::cout << "=== UnitTestHTKCompare1() ===\n";

  std::ifstream is("test_data/test.wav");
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.fbank_htk.1",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use fbank with default configuration...
  FbankOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 0.0;
  op.htk_compat = true;
  op.mel_opts.htk_mode = true;
  op.use_energy = false;  // C0 not energy.

  Fbank fbank(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  fbank.Compute(waveform, 1.0, &kaldi_features, NULL);


  std::cout << "<<<=== Compare with HTK features...\n";
  // compare the results
  bool passed = true;
  int32 i_old = -1;
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      /// THE FEATURES ARE ALMOST IDENTICAL WITH HTK!!! (SEE THE TOLERANCE!)
      if ((std::abs(b - a)) > 0.001) {  //<< TOLERANCE TO DIFFERENCES!!!!!
        // print the non-matching data only once per-line
        if (i_old != i) {
          std::cout << "\n\n\n[HTK-row: " << i << "] " << htk_features.Row(i) << "\n";
          std::cout << "[Kaldi-row: " << i << "] " << kaldi_features.Row(i) << "\n\n\n";
          i_old = i;
        }
        // print indices of non-matching cells
        std::cout << "[" << i << ", " << j << "]";
        passed = false;
  }}}
  if (!passed) KALDI_ERR << "Test failed";

  // write the htk features for later inspection
  HtkHeader header = {
    kaldi_features.NumRows(),
    100000,  // 10ms
    static_cast<int16>(sizeof(float)*kaldi_features.NumCols()),
    000007  // FBANK
  };
  {
    std::ofstream os("tmp.test.wav.fbank_kaldi.1",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fbank_kaldi.1");
}


static void UnitTestHTKCompare2() {
  std::cout << "=== UnitTestHTKCompare2() ===\n";

  std::ifstream is("test_data/test.wav");
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.fbank_htk.2",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use fbank with default configuration...
  FbankOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 25.0;
  op.htk_compat = true;
  op.mel_opts.htk_mode = true;
  op.use_energy = false;  // C0 not energy.

  Fbank fbank(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  fbank.Compute(waveform, 1.0, &kaldi_features, NULL);


  std::cout << "<<<=== Compare with HTK features...\n";
  // compare the results
  bool passed = true;
  int32 i_old = -1;
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      /// THE FEATURES ARE ALMOST IDENTICAL WITH HTK!!! (SEE THE TOLERANCE!)
      if ((std::abs(b - a)) > 0.001) {  //<< TOLERANCE TO DIFFERENCES!!!!!
        // print the non-matching data only once per-line
        if (i_old != i) {
          std::cout << "\n\n\n[HTK-row: " << i << "] " << htk_features.Row(i) << "\n";
          std::cout << "[Kaldi-row: " << i << "] " << kaldi_features.Row(i) << "\n\n\n";
          i_old = i;
        }
        // print indices of non-matching cells
        std::cout << "[" << i << ", " << j << "]";
        passed = false;
  }}}
  if (!passed) KALDI_ERR << "Test failed";

  // write the htk features for later inspection
  HtkHeader header = {
    kaldi_features.NumRows(),
    100000,  // 10ms
    static_cast<int16>(sizeof(float)*kaldi_features.NumCols()),
    000007  // FBANK
  };
  {
    std::ofstream os("tmp.test.wav.fbank_kaldi.1",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fbank_kaldi.1");
}

static void UnitTestHTKCompare3() {
  std::cout << "=== UnitTestHTKCompare3() ===\n";

  std::ifstream is("test_data/test.wav");
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.fbank_htk.3",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use fbank with default configuration...
  FbankOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 25.0;
  op.htk_compat = true;
  op.mel_opts.htk_mode = true;
  op.use_energy = false;  // C0 not energy.

  op.mel_opts.vtln_low = 100.0;
  op.mel_opts.vtln_high = 7500.0;
  BaseFloat vtln_warp = 0.9;

  Fbank fbank(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  fbank.Compute(waveform, vtln_warp, &kaldi_features, NULL);


  std::cout << "<<<=== Compare with HTK features...\n";
  // compare the results
  bool passed = true;
  int32 i_old = -1;
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      /// THE FEATURES ARE ALMOST IDENTICAL WITH HTK!!! (SEE THE TOLERANCE!)
      if ((std::abs(b - a)) > 0.001) {  //<< TOLERANCE TO DIFFERENCES!!!!!
        // print the non-matching data only once per-line
        if (i_old != i) {
          std::cout << "\n\n\n[HTK-row: " << i << "] " << htk_features.Row(i) << "\n";
          std::cout << "[Kaldi-row: " << i << "] " << kaldi_features.Row(i) << "\n\n\n";
          i_old = i;
        }
        // print indices of non-matching cells
        std::cout << "[" << i << ", " << j << "]";
        if (j < 20) passed = false; // We know the last couple of filterbanks differ.  We let this slide.
        else KALDI_WARN << "Ignoring difference in last fbanks, we know the algorithms differ.";
  }}}
  if (!passed) KALDI_ERR << "Test failed";

  // write the htk features for later inspection
  HtkHeader header = {
    kaldi_features.NumRows(),
    100000,  // 10ms
    static_cast<int16>(sizeof(float)*kaldi_features.NumCols()),
    000007  // FBANK
  };
  {
    std::ofstream os("tmp.test.wav.fbank_kaldi.1",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fbank_kaldi.1");
}


static void UnitTestHTKCompare4() {
  std::cout << "=== UnitTestHTKCompare4() ===\n";

  std::ifstream is("test_data/test.wav");
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.fbank_htk.4",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use fbank with default configuration...
  FbankOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 25.0;
  op.htk_compat = true;
  op.mel_opts.htk_mode = true;
  op.use_energy = false;  // C0 not energy.

  op.mel_opts.vtln_low = 100.0;
  op.mel_opts.vtln_high = 7500.0;
  BaseFloat vtln_warp = 1.1;

  Fbank fbank(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  fbank.Compute(waveform, vtln_warp, &kaldi_features, NULL);


  std::cout << "<<<=== Compare with HTK features...\n";
  // compare the results
  bool passed = true;
  int32 i_old = -1;
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      /// THE FEATURES ARE ALMOST IDENTICAL WITH HTK!!! (SEE THE TOLERANCE!)
      if ((std::abs(b - a)) > 0.01) {  //<< TOLERANCE TO DIFFERENCES!!!!!
        // print the non-matching data only once per-line
        if (i_old != i) {
          std::cout << "\n\n\n[HTK-row: " << i << "] " << htk_features.Row(i) << "\n";
          std::cout << "[Kaldi-row: " << i << "] " << kaldi_features.Row(i) << "\n\n\n";
          i_old = i;
        }
        // print indices of non-matching cells
        std::cout << "[" << i << ", " << j << "]";
        passed = false;
  }}}
  if (!passed) KALDI_ERR << "Test failed";

  // write the htk features for later inspection
  HtkHeader header = {
    kaldi_features.NumRows(),
    100000,  // 10ms
    static_cast<int16>(sizeof(float)*kaldi_features.NumCols()),
    000007  // FBANK
  };
  {
    std::ofstream os("tmp.test.wav.fbank_kaldi.1",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fbank_kaldi.1");
}




static void UnitTestFeat() {
  UnitTestReadWave();
  UnitTestSimple();
  UnitTestHTKCompare1();
  UnitTestHTKCompare2();
  UnitTestHTKCompare3();
  UnitTestHTKCompare4();
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



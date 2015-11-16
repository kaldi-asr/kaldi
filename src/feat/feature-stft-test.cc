// feat/feature-stft-test.cc

// Copyright 2015  Hakan Erdogan

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

#include "feat/feature-stft.h"
#include "base/kaldi-math.h"
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"

using namespace kaldi;


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
  StftOptions op;
  // trying to have same opts as baseline.
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "rectangular";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.output_type = "real_and_imaginary";
  op.output_layout = "block";

  Stft stft(op);
  // use default parameters

  // compute stfts.
  stft.Compute(v, &m, NULL);

  // possibly dump
  // std::cout << "== Output features == \n" << m;
  std::cout << "Test passed :)\n\n";
}


static void UnitTestHTKCompare1() {
  std::cout << "=== UnitTestHTKCompare1() ===\n";

  std::ifstream is("test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.stft_htk.1",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use stft with default configuration...
  StftOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.frame_shift_ms = 10.0;
  op.frame_opts.frame_length_ms = 30.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.snip_edges = false;
  op.output_type = "real_and_imaginary";
  op.output_layout = "block";

  Stft stft(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  stft.Compute(waveform, &kaldi_features, NULL);

  // compare the results
  bool passed = true;
  int32 i_old = -1;
  std::cout << "\n HTK-number of rows: " << htk_features.NumRows() << " kaldi number of rows: " << kaldi_features.NumRows() << "\n";
  std::cout << "\n HTK-number of columns: " << htk_features.NumCols() << " kaldi number of columns: " << kaldi_features.NumCols() << "\n";
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      if ((std::abs(b - a)) > 0.1) {  //<< TOLERANCE TO DIFFERENCES!!!!!
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
    021406  // MFCC_D_A_0
  };
  {
    std::ofstream os("tmp.test.wav.stft_kaldi.1",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fea_kaldi.1");
}

static void UnitTestHTKCompare2() {
  std::cout << "=== UnitTestHTKCompare2() ===\n";

  std::ifstream is("test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.stft_htk.2",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use stft with default configuration...
  StftOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.frame_shift_ms = 10.0;
  op.frame_opts.frame_length_ms = 30.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.snip_edges = false;
  op.output_type = "real_and_imaginary";
  op.output_layout = "interleaved";

  Stft stft(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  stft.Compute(waveform, &kaldi_features, NULL);

  // compare the results
  bool passed = true;
  int32 i_old = -1;
  std::cout << "\n HTK-number of rows: " << htk_features.NumRows() << " kaldi number of rows: " << kaldi_features.NumRows() << "\n";
  std::cout << "\n HTK-number of columns: " << htk_features.NumCols() << " kaldi number of columns: " << kaldi_features.NumCols() << "\n";
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      if ((std::abs(b - a)) > 0.1) {  //<< TOLERANCE TO DIFFERENCES!!!!!
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
    021406  // MFCC_D_A_0
  };
  {
    std::ofstream os("tmp.test.wav.stft_kaldi.2",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fea_kaldi.2");
}

static void UnitTestHTKCompare3() {
  std::cout << "=== UnitTestHTKCompare3() ===\n";

  std::ifstream is("test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.stft_htk.3",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use stft with default configuration...
  StftOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.frame_shift_ms = 10.0;
  op.frame_opts.frame_length_ms = 30.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = false;
  op.frame_opts.snip_edges = false;
  op.output_type = "real_and_imaginary";
  op.output_layout = "block";

  Stft stft(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  stft.Compute(waveform, &kaldi_features, NULL);

  // compare the results
  bool passed = true;
  int32 i_old = -1;
  std::cout << "\n HTK-number of rows: " << htk_features.NumRows() << " kaldi number of rows: " << kaldi_features.NumRows() << "\n";
  std::cout << "\n HTK-number of columns: " << htk_features.NumCols() << " kaldi number of columns: " << kaldi_features.NumCols() << "\n";
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      if ((std::abs(b - a)) > 1.0) {  //<< TOLERANCE TO DIFFERENCES!!!!!
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
    021406  // MFCC_D_A_0
  };
  {
    std::ofstream os("tmp.test.wav.stft_kaldi.3",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fea_kaldi.3");
}


static void UnitTestHTKCompare4() {
  std::cout << "=== UnitTestHTKCompare4() ===\n";

  std::ifstream is("test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.stft_htk.4",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use stft with default configuration...
  StftOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.frame_shift_ms = 10.0;
  op.frame_opts.frame_length_ms = 30.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.snip_edges = false;
  op.output_type = "real_and_imaginary";
  op.output_layout = "block";
  op.cut_nyquist = "true";

  Stft stft(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  stft.Compute(waveform, &kaldi_features, NULL);

  // compare the results
  bool passed = true;
  int32 i_old = -1;
  std::cout << "\n HTK-number of rows: " << htk_features.NumRows() << " kaldi number of rows: " << kaldi_features.NumRows() << "\n";
  std::cout << "\n HTK-number of columns: " << htk_features.NumCols() << " kaldi number of columns: " << kaldi_features.NumCols() << "\n";
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      if ((std::abs(b - a)) > 0.1) {  //<< TOLERANCE TO DIFFERENCES!!!!!
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
    021406  // MFCC_D_A_0
  };
  {
    std::ofstream os("tmp.test.wav.stft_kaldi.4",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fea_kaldi.4");
}

static void UnitTestHTKCompare5() {
  std::cout << "=== UnitTestHTKCompare5() ===\n";

  std::ifstream is("test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // read the HTK features
  Matrix<BaseFloat> htk_features;
  {
    std::ifstream is("test_data/test.wav.stft_htk.5",
                     std::ios::in | std::ios_base::binary);
    bool ans = ReadHtk(is, &htk_features, 0);
    KALDI_ASSERT(ans);
  }

  // use stft with default configuration...
  StftOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.frame_shift_ms = 10.0;
  op.frame_opts.frame_length_ms = 30.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.snip_edges = true;
  op.output_type = "real_and_imaginary";
  op.output_layout = "block";

  Stft stft(op);

  // calculate kaldi features
  Matrix<BaseFloat> kaldi_features;
  stft.Compute(waveform, &kaldi_features, NULL);

  // compare the results
  bool passed = true;
  int32 i_old = -1;
  std::cout << "\n HTK-number of rows: " << htk_features.NumRows() << " kaldi number of rows: " << kaldi_features.NumRows() << "\n";
  std::cout << "\n HTK-number of columns: " << htk_features.NumCols() << " kaldi number of columns: " << kaldi_features.NumCols() << "\n";
  KALDI_ASSERT(kaldi_features.NumRows() == htk_features.NumRows());
  KALDI_ASSERT(kaldi_features.NumCols() == htk_features.NumCols());
  // Ignore ends-- we make slightly different choices than
  // HTK about how to treat the deltas at the ends.
  for (int32 i = 10; i+10 < kaldi_features.NumRows(); i++) {
    for (int32 j = 0; j < kaldi_features.NumCols(); j++) {
      BaseFloat a = kaldi_features(i, j), b = htk_features(i, j);
      if ((std::abs(b - a)) > 0.1) {  //<< TOLERANCE TO DIFFERENCES!!!!!
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
    021406  // MFCC_D_A_0
  };
  {
    std::ofstream os("tmp.test.wav.stft_kaldi.5",
                     std::ios::out|std::ios::binary);
    WriteHtk(os, kaldi_features, header);
  }

  std::cout << "Test passed :)\n\n";
  
  unlink("tmp.test.wav.fea_kaldi.5");
}



static void UnitTestFeat() {
  UnitTestSimple();
  UnitTestHTKCompare1();
  UnitTestHTKCompare2();
  UnitTestHTKCompare3();
  UnitTestHTKCompare4();
  UnitTestHTKCompare5();
//  UnitTestHTKCompare6();
  std::cout << "Tests succeeded.\n";
}



int main() {
  try {
  //  for (int i = 0; i < 5; i++)
      UnitTestFeat();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}



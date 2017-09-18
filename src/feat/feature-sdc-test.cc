// feat/feature-sdc-test.cc

// Copyright 2014 David Snyder

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

static void UnitTestCompareWithDeltaFeatures(Matrix<BaseFloat> &raw_features, int32 window) {
  std::cout << "=== UnitTestSDCCompareWithDeltaFeatures() ===\n";
  DeltaFeaturesOptions deltas_opts;
  deltas_opts.window = window;
  ShiftedDeltaFeaturesOptions shifted_deltas_opts;
  shifted_deltas_opts.window = window;
  Matrix<BaseFloat> deltas_features;
  Matrix<BaseFloat> shifted_deltas_features;
  ComputeDeltas(deltas_opts,
                raw_features,
                &deltas_features);
  ComputeShiftedDeltas(shifted_deltas_opts,
                raw_features,
                &shifted_deltas_features);

  int32 dd_num_rows = deltas_features.NumRows();
  int32 sdc_num_rows = shifted_deltas_features.NumRows();
  int32 num_features = raw_features.NumCols();
 
  // Number of rows will be equal, but not
  // columns, in general.
  KALDI_ASSERT(dd_num_rows == sdc_num_rows);

  // The raw mfcc features and the first first-order delta features
  // will be identical in the SDC and Delta-Deltas.
  for (int32 i = 0; i < dd_num_rows; i++) {
    for (int32 j = 0; j < 2 * num_features; j++) {
      BaseFloat a = deltas_features(i, j), b = shifted_deltas_features(i, j);
      KALDI_ASSERT(std::abs(b - a) < 0.001);
    }
  }
}

static void UnitTestParams(Matrix<BaseFloat> &raw_features, int32 window, 
                           int32 shift, int32 n_blocks) {
  std::cout << "=== UnitTestSDCParams() ===\n";
  ShiftedDeltaFeaturesOptions shifted_deltas_opts;
  shifted_deltas_opts.window = window;
  shifted_deltas_opts.num_blocks = n_blocks;
  shifted_deltas_opts.block_shift = shift;

  Matrix<BaseFloat> shifted_deltas_features;
  ComputeShiftedDeltas(shifted_deltas_opts,
                raw_features,
                &shifted_deltas_features);

  int32 raw_num_cols = raw_features.NumCols();
  int32 sdc_num_rows = shifted_deltas_features.NumRows();
  int32 sdc_num_cols = shifted_deltas_features.NumCols();

  KALDI_ASSERT(sdc_num_cols == raw_num_cols * (n_blocks  + 1));
  
  /* For every coefficient in the raw feature vector a 
     delta is calculated and appended to the new feature vector,
     as is done normally in a delta-deltas computation.
     In addition, n_blocks delta in advance are also appended.
     Somewhere in advance of the current position, say at
     t + l these additional delta are the first order deltas
     at that position (t + l). The following code works out a
     mapping from these additional deltas to where they would
     appear in a delta-deltas computation and verfies these
     values' equality. */
  for (int32 i = 0; i < sdc_num_rows; i++) { 
    for (int32 j = 2 * raw_num_cols; j < sdc_num_cols; j += raw_num_cols) {
      for (int32 k = 0; k < raw_num_cols; k++) {
        int32 row = i + (j/raw_num_cols - 1) * shift;
        if (row < sdc_num_rows) {
          BaseFloat a = shifted_deltas_features(i, j + k);
          BaseFloat b = shifted_deltas_features(row, raw_num_cols + k);
          KALDI_ASSERT(std::abs(a - b) < 0.001);
        }
      }
    }
  }
}

static void UnitTestEndEffects(Matrix<BaseFloat> &raw_features, int32 window, 
                               int32 shift, int32 n_blocks) {
  std::cout << "=== UnitTestSDCEndEffects() ===\n";
  ShiftedDeltaFeaturesOptions shifted_deltas_opts;
  shifted_deltas_opts.window = window;
  shifted_deltas_opts.num_blocks = n_blocks;
  shifted_deltas_opts.block_shift = shift;

  Matrix<BaseFloat> shifted_deltas_features;
  ComputeShiftedDeltas(shifted_deltas_opts,
                raw_features,
                &shifted_deltas_features);
  int32 raw_num_cols = raw_features.NumCols();
  int32 sdc_num_rows = shifted_deltas_features.NumRows();
  int32 sdc_num_cols = shifted_deltas_features.NumCols();
  
  // If the entire window is out-of-bounds the delta should be zero.
  for (int32 i = sdc_num_rows - n_blocks + 1; i < sdc_num_rows; i++) {
    for (int32 j = 2 * raw_num_cols; j < sdc_num_cols; j += raw_num_cols) {
      for (int32 k = 0; k < raw_num_cols; k++) {
        if (i + (j/raw_num_cols - 1) * shift - window/2 > sdc_num_rows)
          KALDI_ASSERT(shifted_deltas_features(i, j + k) <= 0.00001);
      }
    } 
  }
}

int main() {
  std::ifstream is("test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // mfcc with default configuration...
  MfccOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 0.0;
  op.use_energy = false;
  Mfcc mfcc(op);
  Matrix<BaseFloat> raw_features;
  mfcc.Compute(waveform, 1.0, &raw_features);

  try {
    for (int32 window = 1; window < 4; window++) {
      UnitTestCompareWithDeltaFeatures(raw_features, window);
      for (int32 shift = 1; shift < 10; shift++) {
        for (int32 n_blocks = 1; n_blocks < 20; n_blocks += 3) {
          UnitTestParams(raw_features, window, shift, n_blocks);
          UnitTestEndEffects(raw_features, window, shift, n_blocks);
        }
      }
    }
    return 0;
  } catch (const std::exception &e) {
    static_cast<void>(e);
    return 1;
  }
  
}


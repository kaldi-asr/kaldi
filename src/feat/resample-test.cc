// feat/resample-test.cc

// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
//              2014  Yanqing Sun, Junjie Wang
//              2014  Johns Hopkins University (author: Daniel Povey)

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


#include "feat/resample.h"

using namespace kaldi;

class TestFunction {
 public:
  explicit TestFunction(double frequency):
      frequency_(frequency),
      sin_magnitude_(RandGauss()),
      cos_magnitude_(RandGauss()) { }

  double operator() (double t) const {
    double omega_t = t * M_2PI * frequency_;
    return sin_magnitude_ * sin(omega_t)
        + cos_magnitude_ * cos(omega_t);
  }
 private:
  double frequency_;
  double sin_magnitude_;
  double cos_magnitude_;
};


void UnitTestArbitraryResample() {
  BaseFloat samp_freq = 1000.0 * (1.0 + RandUniform());
  int32 num_samp = 256 + static_cast<int32>((RandUniform() * 256));

  BaseFloat time_interval = num_samp / samp_freq;

  // Choose a lowpass frequency that's lower than 95% of the Nyquist.
  BaseFloat lowpass_freq = samp_freq * 0.95 * 0.5 / (1.0 + RandUniform());

  // Number of zeros of the sinc function that the window extends out to.
  int32 num_zeros = 3 + rand() % 10;

  // Resample the signal at arbitrary points within that time interval.
  int32 num_resamp = 50 + rand() % 100;  // Resample at around 100 points,
                                         // anywhere in the signal.


  Vector<BaseFloat> resample_points(num_resamp);
  for (int32 i = 0; i < num_resamp; i++) {
    // the if-statement is to make some of the resample_points
    // exactly coincide with the original points, to activate
    // a certain code path.
    if (rand() % 2 == 0)
      resample_points(i) = (rand() % num_samp) / samp_freq;
    else
      resample_points(i) = RandUniform() * time_interval;
  }



  BaseFloat window_width = num_zeros / (2.0 * lowpass_freq);
  // the resampling should be quite accurate if we are further
  // than filter_width away from the edges.
  BaseFloat min_t = 0.0 + window_width,
      max_t = time_interval - (1.0 / samp_freq) - window_width;

  // window_freq gives us a rough idea of the frequency spread
  // that the windowing function gives us; we want the test frequency
  // to be lower than the lowpass frequency by at least this much.
  // (note: the real width of the window from side to side
  // is 2.0 * window_width)
  BaseFloat window_freq = 1.0 / (2.0 * window_width),
      freq_margin = 2.0 * window_freq;

  // Choose a test-signal frequency that's lower than
  // lowpass_freq - freq_margin.
  BaseFloat test_signal_freq =
    (lowpass_freq - freq_margin) * (1.0 / (1.0 + RandUniform()));

  KALDI_ASSERT(test_signal_freq > 0.0);

  ArbitraryResample resampler(num_samp, samp_freq, lowpass_freq,
                              resample_points, num_zeros);


  TestFunction test_func(test_signal_freq);

  // test with a one-row matrix equal to the test signal.
  Matrix<BaseFloat> sample_values(1, num_samp);
  for (int32 i = 0; i < num_samp; i++) {
    BaseFloat t = i / samp_freq;
    sample_values(0, i) = test_func(t);
  }
  Matrix<BaseFloat> resampled_values(1, num_resamp);


  if (rand() % 2 == 0) {
    resampler.Resample(sample_values,
                       &resampled_values);
  } else {
    SubVector<BaseFloat> out(resampled_values, 0);
    resampler.Resample(sample_values.Row(0),
                       &out);
  }


  for (int32 i = 0; i < num_resamp; i++) {
    BaseFloat t = resample_points(i),
        x1 = test_func(t),
        x2 = resampled_values(0, i),
        error = fabs(x1 - x2);
    if (i % 10 == 0) {
      KALDI_VLOG(1) << "Error is " << error << ", t = " << t
                << ", samp_freq = " << samp_freq << ", lowpass_freq = "
                << lowpass_freq << ", test_freq = " << test_signal_freq
                << ", num-zeros is " << num_zeros;
    }
    if (t > min_t && t < max_t) {
      if (num_zeros == 3) {
        KALDI_ASSERT(error < 0.1);
      } else {
        KALDI_ASSERT(error < 0.025);
      }
    } else {
      KALDI_VLOG(1) << "[not checking since out of bounds]";
    }
  }
}


void UnitTestLinearResample() {
  // this test makes sure that LinearResample gives identical results to
  // ArbitraryResample when set up the same way, even if the signal is broken up
  // into many pieces.

  int32 samp_freq = 1000.0 * (1.0 + RandUniform()),
      resamp_freq = 1000.0 * (1.0 + RandUniform());
  // note: these are both integers!
  int32 num_samp = 256 + static_cast<int32>((RandUniform() * 256));

  BaseFloat time_interval = num_samp / static_cast<BaseFloat>(samp_freq);

  // Choose a lowpass frequency that's lower than 95% of the Nyquist of both
  // of the frequencies..
  BaseFloat lowpass_freq =
    std::min(samp_freq, resamp_freq) * 0.95 * 0.5 / (1.0 + RandUniform());

  // Number of zeros of the sinc function that the window extends out to.
  int32 num_zeros = 3 + rand() % 10;

  // compute the number of "resample" points.
  int32 num_resamp = ceil(time_interval * resamp_freq);

  Vector<BaseFloat> resample_points(num_resamp);
  for (int32 i = 0; i < num_resamp; i++)
    resample_points(i) = i / static_cast<BaseFloat>(resamp_freq);


  Vector<BaseFloat> test_signal(num_samp);
  test_signal.SetRandn();

  ArbitraryResample resampler(num_samp, samp_freq, lowpass_freq,
                              resample_points, num_zeros);


  // test with a one-row matrix equal to the test signal.
  Matrix<BaseFloat> sample_values(1, num_samp);
  sample_values.Row(0).CopyFromVec(test_signal);

  Matrix<BaseFloat> resampled_values(1, num_resamp);

  resampler.Resample(sample_values,
                     &resampled_values);

  LinearResample linear_resampler(samp_freq, resamp_freq,
                                  lowpass_freq, num_zeros);

  Vector<BaseFloat> resampled_vec;

  linear_resampler.Resample(test_signal, true, &resampled_vec);

  if (!ApproxEqual(resampled_values.Row(0), resampled_vec)) {
    KALDI_LOG << "ArbitraryResample: " << resampled_values.Row(0);
    KALDI_LOG << "LinearResample: " << resampled_vec;
    KALDI_ERR << "Signals differ.";
  }

  // Check it gives the same results when the input is broken up into pieces.
  Vector<BaseFloat> resampled_vec2;
  int32 input_dim_seen = 0;
  while (input_dim_seen < test_signal.Dim()) {
    int32 dim_remaining = test_signal.Dim() - input_dim_seen;
    int32 piece_size = rand() % std::min(dim_remaining + 1, 10);
    KALDI_VLOG(1) << "Piece size = " << piece_size;
    SubVector<BaseFloat> in_piece(test_signal, input_dim_seen, piece_size);
    Vector<BaseFloat> out_piece;
    bool flush = (piece_size == dim_remaining);
    linear_resampler.Resample(in_piece, flush, &out_piece);
    int32 old_output_dim = resampled_vec2.Dim();
    resampled_vec2.Resize(old_output_dim + out_piece.Dim(), kCopyData);
    resampled_vec2.Range(old_output_dim, out_piece.Dim())
                  .CopyFromVec(out_piece);
    input_dim_seen += piece_size;
  }

  if (!ApproxEqual(resampled_values.Row(0), resampled_vec2)) {
    KALDI_LOG << "ArbitraryResample: " << resampled_values.Row(0);
    KALDI_LOG << "LinearResample[broken-up]: " << resampled_vec2;
    KALDI_ERR << "Signals differ.";
  }
}

void UnitTestLinearResample2() {
  int32 num_samp = 150 + rand() % 100;
  BaseFloat samp_freq = 1000, resamp_freq = 4000;

  int32 num_zeros = 10; // fairly accurate.
  Vector<BaseFloat> signal_orig(num_samp);
  signal_orig.SetRandn();

  Vector<BaseFloat> signal(num_samp);  
  { // make sure signal is sufficiently low pass, i.e. that we have enough
    // headroom before the Nyquist.
    LinearResample linear_resampler_filter(samp_freq, samp_freq,
                                           0.8 * samp_freq / 2.0, num_zeros);
    linear_resampler_filter.Resample(signal_orig, true, &signal);
  }
  

  Vector<BaseFloat> signal_upsampled;

  LinearResample linear_resampler(samp_freq, resamp_freq,
                                  samp_freq / 2.0, num_zeros);

  linear_resampler.Resample(signal, true, &signal_upsampled);

  // resample back to the original frequency.
  LinearResample linear_resampler2(resamp_freq, samp_freq,
                                   samp_freq / 2.0, num_zeros);
  
  
  Vector<BaseFloat> signal_downsampled;  
  linear_resampler2.Resample(signal_upsampled, true, &signal_downsampled);


  int32 samp_discard = 30;  // Discard 20 samples for edge effects.
  SubVector<BaseFloat> signal_middle(signal, samp_discard,
                                     signal.Dim() - (2 * samp_discard));

  SubVector<BaseFloat> signal2_middle(signal_downsampled, samp_discard,
                                      signal.Dim() - (2 * samp_discard));

  BaseFloat self1 = VecVec(signal_middle, signal_middle),
      self2 = VecVec(signal2_middle, signal2_middle),
      cross = VecVec(signal_middle, signal2_middle);
  KALDI_LOG << "Self1 = " << self1 << ", self2 = " << self2
            << ", cross = " << cross;
  AssertEqual(self1, self2, 0.001);
  AssertEqual(self1, cross, 0.001);
}

int main() {
  try {
    for (int32 x = 0; x < 50; x++)
      UnitTestLinearResample();
    for (int32 x = 0; x < 50; x++)
      UnitTestLinearResample2();    
    for (int32 x = 0; x < 50; x++)
      UnitTestArbitraryResample();

    KALDI_LOG << "Tests succeeded.\n";
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return 1;
  }
}

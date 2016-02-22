// feat/feature-spectrogram.cc

// Copyright 2009-2012  Karel Vesely
// Copyright 2012  Navdeep Jaitly

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


#include "feat/feature-spectrogram.h"


namespace kaldi {

Spectrogram::Spectrogram(const SpectrogramOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 num_fft_bins = opts_.frame_opts.NumFftBins();
  if ((num_fft_bins & (num_fft_bins-1)) == 0)  // Is a power of two
    srfft_ = new SplitRadixRealFft<BaseFloat>(num_fft_bins);
}

Spectrogram::~Spectrogram() {
  delete srfft_;
}

void Spectrogram::Compute(const VectorBase<BaseFloat> &wave,
                          Matrix<BaseFloat> *output,
                          Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);

  int32 num_fft_bins = opts_.frame_opts.NumFftBins();
 
  BaseFloat sample_freq = opts_.frame_opts.samp_freq;
  BaseFloat nyquist = 0.5 * sample_freq;
  BaseFloat low_freq = opts_.low_freq, high_freq;
  if (opts_.high_freq > 0.0)
    high_freq = opts_.high_freq;
  else
    high_freq = nyquist + opts_.high_freq;
  
  if (low_freq < 0.0 || low_freq >= nyquist
      || high_freq <= 0.0 || high_freq > nyquist
      || high_freq <= low_freq)
    KALDI_ERR << "Bad values in options: low-freq " << low_freq
              << " and high-freq " << high_freq << " vs. nyquist "
              << nyquist;
    
  int32 low_c = low_freq / sample_freq * num_fft_bins;
  int32 high_c = high_freq / sample_freq * num_fft_bins;

  int32 cols_out =  high_c - low_c + 1;

  if (opts_.use_energy && low_c != 0) {
    cols_out++;
  }

  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  // Prepare the output buffer
  output->Resize(rows_out, cols_out);

  // Optionally extract the remainder for further processing
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  BaseFloat log_energy;

  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    // Cut the window, apply window function
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &window, (opts_.use_energy && opts_.raw_energy ? &log_energy : NULL));
    
    KALDI_ASSERT(window.Dim() <= num_fft_bins);
    window.Resize(num_fft_bins, kCopyData);

    // Compute energy after window function (not the raw one)
    if (opts_.use_energy && !opts_.raw_energy)
      log_energy = Log(std::max(VecVec(window, window),
                                std::numeric_limits<BaseFloat>::min()));
    
    if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
      srfft_->Compute(window.Data(), true);
    else  // An alternative algorithm that works for non-powers-of-two
      RealFft(&window, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);

    power_spectrum.ApplyFloor(std::numeric_limits<BaseFloat>::min());
    power_spectrum.ApplyLog();

    // Output buffers
    SubVector<BaseFloat> this_output(
        (output->Row(r)).Range((opts_.use_energy && low_c != 0) ? 1 : 0, high_c - low_c + 1));
    SubVector<BaseFloat> this_power_spectrum(power_spectrum, 
                                             low_c, high_c - low_c + 1);
    this_output.CopyFromVec(this_power_spectrum);
    if (opts_.use_energy && opts_.energy_floor > 0.0 && log_energy < log_energy_floor_) {
        log_energy = log_energy_floor_;
    }
    if (opts_.use_energy)
      this_output(0) = log_energy;
  }
}

}  // namespace kaldi

// feat/feature-fbank.cc

// Copyright 2009-2012  Karel Vesely

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


#include "feat/feature-fbank.h"


namespace kaldi {

Fbank::Fbank(const FbankOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Fbank::~Fbank() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;
  if (srfft_ != NULL)
    delete srfft_;
}

const MelBanks *Fbank::GetMelBanks(BaseFloat vtln_warp) {
  MelBanks *this_mel_banks = NULL;
  std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts,
                                  opts_.frame_opts,
                                  vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}

void Fbank::Compute(const VectorBase<BaseFloat> &wave,
                   BaseFloat vtln_warp,
                   Matrix<BaseFloat> *output,
                   Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
  int32 cols_out = opts_.mel_opts.num_bins + opts_.use_energy;
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  // Prepare the output buffer
  output->Resize(rows_out, cols_out);

  // Optionally extract the remainder for further processing
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  Vector<BaseFloat> mel_energies;
  BaseFloat log_energy;

  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    // Cut the window, apply window function
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_, &window,
                  (opts_.use_energy && opts_.raw_energy ? &log_energy : NULL));

    // Compute energy after window function (not the raw one)
    if (opts_.use_energy && !opts_.raw_energy)
      log_energy = log(VecVec(window, window));

    if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
      srfft_->Compute(window.Data(), true);
    else  // An alternative algorithm that works for non-powers-of-two.
      RealFft(&window, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);

    // Integrate with MelFiterbank over power spectrum
    const MelBanks *this_mel_banks = GetMelBanks(vtln_warp);
    this_mel_banks->Compute(power_spectrum, &mel_energies);
    if (opts_.use_log_fbank)
      mel_energies.ApplyLog();  // take the log.

    // Output buffers
    SubVector<BaseFloat> this_output(output->Row(r));
    SubVector<BaseFloat> this_fbank(this_output.Range((opts_.use_energy? 1 : 0),
                                                      opts_.mel_opts.num_bins));

    // Copy to output
    this_fbank.CopyFromVec(mel_energies);
    // Copy energy as first value
    if (opts_.use_energy) {
      if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_) {
        log_energy = log_energy_floor_;
      }
      this_output(0) = log_energy;
    }

    // HTK compat: Shift features, so energy is last value
    if (opts_.htk_compat && opts_.use_energy) {
      BaseFloat energy = this_output(0);
      for (int32 i = 0; i < opts_.mel_opts.num_bins; i++) {
        this_output(i) = this_output(i+1);
      }
      this_output(opts_.mel_opts.num_bins) = energy;
    }
  }
}

}  // namespace kaldi

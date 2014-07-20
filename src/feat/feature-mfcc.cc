// feat/feature-mfcc.cc

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


#include "feat/feature-mfcc.h"


namespace kaldi {

Mfcc::Mfcc(const MfccOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {
  int32 num_bins = opts.mel_opts.num_bins;
  Matrix<BaseFloat> dct_matrix(num_bins, num_bins);
  ComputeDctMatrix(&dct_matrix);
  // Note that we include zeroth dct in either case.  If using the
  // energy we replace this with the energy.  This means a different
  // ordering of features than HTK.
  SubMatrix<BaseFloat> dct_rows(dct_matrix, 0, opts.num_ceps, 0, num_bins);
  dct_matrix_.Resize(opts.num_ceps, num_bins);
  dct_matrix_.CopyFromMat(dct_rows);  // subset of rows.
  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_.Resize(opts.num_ceps);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
  }
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
  
  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]  The reason we call this here is to
  // improve the efficiency of the "const" version of Compute().
  GetMelBanks(1.0);
}

Mfcc::~Mfcc() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;
  if (srfft_ != NULL)
    delete srfft_;
}

const MelBanks *Mfcc::GetMelBanks(BaseFloat vtln_warp) {
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


const MelBanks *Mfcc::GetMelBanks(BaseFloat vtln_warp, bool *must_delete) const {
  MelBanks *this_mel_banks = NULL;
  std::map<BaseFloat, MelBanks*>::const_iterator iter =
      mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts,
                                  opts_.frame_opts,
                                  vtln_warp);
    *must_delete = true;
  } else {
    this_mel_banks = iter->second;
    *must_delete = false;
  }
  return this_mel_banks;
}


void Mfcc::Compute(const VectorBase<BaseFloat> &wave,
                   BaseFloat vtln_warp,
                   Matrix<BaseFloat> *output,
                   Vector<BaseFloat> *wave_remainder) {
  const MelBanks *this_mel_banks = GetMelBanks(vtln_warp);
  ComputeInternal(wave, *this_mel_banks, output, wave_remainder);  
}

void Mfcc::Compute(const VectorBase<BaseFloat> &wave,
                   BaseFloat vtln_warp,
                   Matrix<BaseFloat> *output,
                   Vector<BaseFloat> *wave_remainder) const {
  bool must_delete_mel_banks;
  const MelBanks *mel_banks = GetMelBanks(vtln_warp,
                                               &must_delete_mel_banks);
  
  ComputeInternal(wave, *mel_banks, output, wave_remainder);
  
  if (must_delete_mel_banks)
    delete mel_banks;
}

void Mfcc::ComputeInternal(const VectorBase<BaseFloat> &wave,
                           const MelBanks &mel_banks,
                           Matrix<BaseFloat> *output,
                           Vector<BaseFloat> *wave_remainder) const {
  KALDI_ASSERT(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts),
      cols_out = opts_.num_ceps;
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  output->Resize(rows_out, cols_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  Vector<BaseFloat> window;  // windowed waveform.
  Vector<BaseFloat> mel_energies;
  std::vector<BaseFloat> temp_buffer;  // used by srfft.
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index..
    BaseFloat log_energy;
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_, &window,
                  (opts_.use_energy && opts_.raw_energy ? &log_energy : NULL));

    if (opts_.use_energy && !opts_.raw_energy)
      log_energy = log(std::max(VecVec(window, window),
                                std::numeric_limits<BaseFloat>::min()));

    if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
      srfft_->Compute(window.Data(), true, &temp_buffer);
    else  // An alternative algorithm that works for non-powers-of-two.
      RealFft(&window, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);

    mel_banks.Compute(power_spectrum, &mel_energies);

    // avoid log of zero (which should be prevented anyway by dithering).
    mel_energies.ApplyFloor(std::numeric_limits<BaseFloat>::min());
    mel_energies.ApplyLog();  // take the log.

    SubVector<BaseFloat> this_mfcc(output->Row(r));

    // this_mfcc = dct_matrix_ * mel_energies [which now have log]
    this_mfcc.AddMatVec(1.0, dct_matrix_, kNoTrans, mel_energies, 0.0);

    if (opts_.cepstral_lifter != 0.0)
      this_mfcc.MulElements(lifter_coeffs_);

    if (opts_.use_energy) {
      if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_)
        log_energy = log_energy_floor_;
      this_mfcc(0) = log_energy;
    }

    if (opts_.htk_compat) {
      BaseFloat energy = this_mfcc(0);
      for (int32 i = 0; i < opts_.num_ceps-1; i++)
        this_mfcc(i) = this_mfcc(i+1);
      if (!opts_.use_energy)
        energy *= M_SQRT2;  // scale on C0 (actually removing scale
      // we previously added that's part of one common definition of
      // cosine transform.)
      this_mfcc(opts_.num_ceps-1)  = energy;
    }
  }
}






}  // namespace kaldi

// feat/feature-plp.cc

// Copyright 2009-2011  Petr Motlicek;  Karel Vesely

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


#include "feat/feature-plp.h"
#include "util/parse-options.h"


namespace kaldi {

Plp::Plp(const PlpOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts), srfft_(NULL) {
  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_.Resize(opts.num_ceps);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
  }
  InitIdftBases(opts_.lpc_order + 1, opts_.mel_opts.num_bins + 2,
                &idft_bases_);

  if (opts.energy_floor > 0.0)
    log_energy_floor_ = log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Plp::~Plp() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;

  for (std::map<BaseFloat,
                Vector<BaseFloat>* >::iterator iter = equal_loudness_.begin();
      iter != equal_loudness_.end();
      ++iter)
    delete iter->second;

  if (srfft_ != NULL)
    delete srfft_;
}

const MelBanks *Plp::GetMelBanks(BaseFloat vtln_warp) {
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

const Vector<BaseFloat> *Plp::GetEqualLoudness(BaseFloat vtln_warp) {
  const MelBanks *this_mel_banks = GetMelBanks(vtln_warp);
  Vector<BaseFloat> *ans = NULL;
  std::map<BaseFloat, Vector<BaseFloat>*>::iterator iter
      = equal_loudness_.find(vtln_warp);
  if (iter == equal_loudness_.end()) {
    ans = new Vector<BaseFloat>;
    GetEqualLoudnessVector(*this_mel_banks, ans);
    equal_loudness_[vtln_warp] = ans;
  } else {
    ans = iter->second;
  }
  return ans;
}

void Plp::Compute(const VectorBase<BaseFloat> &wave,
                  BaseFloat vtln_warp,
                  Matrix<BaseFloat> *output,
                  Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts),
      cols_out = opts_.num_ceps;
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  output->Resize(rows_out, cols_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  Vector<BaseFloat> window;  // windowed waveform.
  int32 num_mel_bins = opts_.mel_opts.num_bins;
  Vector<BaseFloat> mel_energies(num_mel_bins);
  Vector<BaseFloat> mel_energies_duplicated(num_mel_bins+2);
  Vector<BaseFloat> autocorr_coeffs(opts_.lpc_order+1);
  Vector<BaseFloat> lpc_coeffs(opts_.lpc_order);
  Vector<BaseFloat> raw_cepstrum(opts_.lpc_order);  // not including C0,
  // and size may differ from final size.
  Vector<BaseFloat> final_cepstrum(opts_.num_ceps);
  KALDI_ASSERT(opts_.num_ceps <= opts_.lpc_order+1);  // our num-ceps includes C0.
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index..
    BaseFloat log_energy;
    ExtractWindow(wave, r, opts_.frame_opts,
                  feature_window_function_, &window,
                  (opts_.use_energy && opts_.raw_energy ? &log_energy : NULL));

    if (opts_.use_energy && !opts_.raw_energy)
      log_energy = log(VecVec(window, window));

    if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
      srfft_->Compute(window.Data(), true);
    else  // An alternative algorithm that works for non-powers-of-two.
      RealFft(&window, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);  // elements 0 ... window.Dim()/2

    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);

    const MelBanks *this_mel_banks = GetMelBanks(vtln_warp);

    this_mel_banks->Compute(power_spectrum, &mel_energies);

    // HTK doesn't log the mel bank outputs for the PLPs' [HARDCODED]
    // mel_energies.ApplyLog();  // take the log.

    mel_energies.MulElements(*GetEqualLoudness(vtln_warp));

    mel_energies.ApplyPow(opts_.compress_factor);

    // duplicate first and last elements.
    {
      SubVector<BaseFloat> v(mel_energies_duplicated, 1, num_mel_bins);
      v.CopyFromVec(mel_energies);
    }
    mel_energies_duplicated(0) = mel_energies(0);
    mel_energies_duplicated(num_mel_bins+1) = mel_energies(num_mel_bins-1);

    autocorr_coeffs.AddMatVec(1.0, idft_bases_, kNoTrans,
                              mel_energies_duplicated,  0.0);

    BaseFloat energy = ComputeLpc(autocorr_coeffs, &lpc_coeffs);

    Lpc2Cepstrum(opts_.lpc_order, lpc_coeffs.Data(), raw_cepstrum.Data());
    {
      SubVector<BaseFloat> dst(final_cepstrum, 1, opts_.num_ceps-1);
      SubVector<BaseFloat> src(raw_cepstrum, 0, opts_.num_ceps-1);
      dst.CopyFromVec(src);
      final_cepstrum(0) = energy;
    }

    if (opts_.cepstral_lifter != 0.0)
      final_cepstrum.MulElements(lifter_coeffs_);

    if (opts_.cepstral_scale != 1.0)
      final_cepstrum.Scale(opts_.cepstral_scale);

    if (opts_.use_energy) {
      if (opts_.energy_floor > 0.0 && log_energy < log_energy_floor_)
        log_energy = log_energy_floor_;
      final_cepstrum(0) = log_energy;
    }

    if (opts_.htk_compat) {
      BaseFloat energy = final_cepstrum(0);
      for (int32 i = 0; i < opts_.num_ceps-1; i++)
        final_cepstrum(i) = final_cepstrum(i+1);
      // if (!opts_.use_energy)
        // energy *= M_SQRT2;  // scale on C0 (actually removing scale
      // we previously added that's part of one common definition of
      // cosine transform.)
      final_cepstrum(opts_.num_ceps-1)  = energy;
    }

    output->Row(r).CopyFromVec(final_cepstrum);
    // std::cout << "FIN" << final_cepstrum;
  }
}


}  // namespace kaldi

// feat/feature-plp.cc

// Copyright 2009-2011  Petr Motlicek;  Karel Vesely
//                2016  Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {

PlpComputer::PlpComputer(const PlpOptions &opts):
    opts_(opts), srfft_(NULL),
    mel_energies_duplicated_(opts_.mel_opts.num_bins + 2, kUndefined),
    autocorr_coeffs_(opts_.lpc_order + 1, kUndefined),
    lpc_coeffs_(opts_.lpc_order, kUndefined),
    raw_cepstrum_(opts_.lpc_order, kUndefined) {

  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_.Resize(opts.num_ceps);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
  }
  InitIdftBases(opts_.lpc_order + 1, opts_.mel_opts.num_bins + 2,
                &idft_bases_);

  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

PlpComputer::PlpComputer(const PlpComputer &other):
    opts_(other.opts_), lifter_coeffs_(other.lifter_coeffs_),
    idft_bases_(other.idft_bases_), log_energy_floor_(other.log_energy_floor_),
    mel_banks_(other.mel_banks_), equal_loudness_(other.equal_loudness_),
    srfft_(NULL),
    mel_energies_duplicated_(opts_.mel_opts.num_bins + 2, kUndefined),
    autocorr_coeffs_(opts_.lpc_order + 1, kUndefined),
    lpc_coeffs_(opts_.lpc_order, kUndefined),
    raw_cepstrum_(opts_.lpc_order, kUndefined) {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
       iter != mel_banks_.end(); ++iter)
    iter->second = new MelBanks(*(iter->second));
  for (std::map<BaseFloat, Vector<BaseFloat>*>::iterator
           iter = equal_loudness_.begin();
       iter != equal_loudness_.end(); ++iter)
    iter->second = new Vector<BaseFloat>(*(iter->second));
  if (other.srfft_ != NULL)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*(other.srfft_));
}

PlpComputer::~PlpComputer() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end(); ++iter)
    delete iter->second;
  for (std::map<BaseFloat, Vector<BaseFloat>* >::iterator
           iter = equal_loudness_.begin();
       iter != equal_loudness_.end(); ++iter)
    delete iter->second;
  delete srfft_;
}

const MelBanks *PlpComputer::GetMelBanks(BaseFloat vtln_warp) {
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

const Vector<BaseFloat> *PlpComputer::GetEqualLoudness(BaseFloat vtln_warp) {
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

void PlpComputer::Compute(BaseFloat signal_raw_log_energy,
                          BaseFloat vtln_warp,
                          VectorBase<BaseFloat> *signal_frame,
                          VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());

  const MelBanks &mel_banks = *GetMelBanks(vtln_warp);
  const Vector<BaseFloat> &equal_loudness = *GetEqualLoudness(vtln_warp);


  KALDI_ASSERT(opts_.num_ceps <= opts_.lpc_order+1);  // our num-ceps includes C0.


  if (opts_.use_energy && !opts_.raw_energy)
    signal_raw_log_energy = Log(std::max<BaseFloat>(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<float>::min()));

  if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(signal_frame, true);

  // Convert the FFT into a power spectrum.
  ComputePowerSpectrum(signal_frame);  // elements 0 ... signal_frame->Dim()/2

  SubVector<BaseFloat> power_spectrum(*signal_frame,
                                      0, signal_frame->Dim() / 2 + 1);

  int32 num_mel_bins = opts_.mel_opts.num_bins;

  SubVector<BaseFloat> mel_energies(mel_energies_duplicated_, 1, num_mel_bins);

  mel_banks.Compute(power_spectrum, &mel_energies);

  mel_energies.MulElements(equal_loudness);

  mel_energies.ApplyPow(opts_.compress_factor);

  // duplicate first and last elements
  mel_energies_duplicated_(0) = mel_energies_duplicated_(1);
  mel_energies_duplicated_(num_mel_bins + 1) =
      mel_energies_duplicated_(num_mel_bins);

  autocorr_coeffs_.SetZero();  // In case of NaNs or infs
  autocorr_coeffs_.AddMatVec(1.0, idft_bases_, kNoTrans,
                             mel_energies_duplicated_,  0.0);

  BaseFloat residual_log_energy = ComputeLpc(autocorr_coeffs_, &lpc_coeffs_);

  residual_log_energy = std::max<BaseFloat>(residual_log_energy,
                                 std::numeric_limits<float>::min());

  Lpc2Cepstrum(opts_.lpc_order, lpc_coeffs_.Data(), raw_cepstrum_.Data());
  feature->Range(1, opts_.num_ceps - 1).CopyFromVec(
      raw_cepstrum_.Range(0, opts_.num_ceps - 1));
  (*feature)(0) = residual_log_energy;

  if (opts_.cepstral_lifter != 0.0)
    feature->MulElements(lifter_coeffs_);

  if (opts_.cepstral_scale != 1.0)
    feature->Scale(opts_.cepstral_scale);

  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0 && signal_raw_log_energy < log_energy_floor_)
      signal_raw_log_energy = log_energy_floor_;
    (*feature)(0) = signal_raw_log_energy;
  }

  if (opts_.htk_compat) {  // reorder the features.
    BaseFloat log_energy = (*feature)(0);
    for (int32 i = 0; i < opts_.num_ceps-1; i++)
      (*feature)(i) = (*feature)(i+1);
    (*feature)(opts_.num_ceps-1)  = log_energy;
  }
}


}  // namespace kaldi

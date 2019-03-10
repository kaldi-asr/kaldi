// feat/feature-fbank.cc

// Copyright 2009-2012  Karel Vesely
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


#include "feat/feature-fbank.h"

namespace kaldi {

FbankComputer::FbankComputer(const FbankOptions &opts):
    opts_(opts), srfft_(NULL) {
  KALDI_ASSERT(opts.energy_floor > 0.0 && "Nonzero energy floor is required.");

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

FbankComputer::FbankComputer(const FbankComputer &other):
    opts_(other.opts_),
    mel_banks_(other.mel_banks_), srfft_(NULL) {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    iter->second = new MelBanks(*(iter->second));
  if (other.srfft_)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*(other.srfft_));
}

FbankComputer::~FbankComputer() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end(); ++iter)
    delete iter->second;
  delete srfft_;
}

const MelBanks* FbankComputer::GetMelBanks(BaseFloat vtln_warp) {
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

void FbankComputer::Compute(BaseFloat vtln_warp,
                            VectorBase<BaseFloat> *signal_frame,
                            VectorBase<BaseFloat> *feature) {

  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());


  BaseFloat signal_log_energy;
  if (opts_.use_energy)
    signal_log_energy = Log(std::max<BaseFloat>(VecVec(*signal_frame, *signal_frame),
                                                opts_.energy_floor));

  if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(signal_frame, true);

  // Convert the FFT into a power spectrum.
  ComputePowerSpectrum(signal_frame);
  SubVector<BaseFloat> power_spectrum(*signal_frame, 0,
                                      signal_frame->Dim() / 2 + 1);

  int32 mel_offset = (opts_.use_energy ? 1 : 0);
  SubVector<BaseFloat> mel_energies(*feature,
                                    mel_offset,
                                    opts_.mel_opts.num_bins);

  // Sum with mel fiterbanks over the power spectrum
  mel_banks.Compute(power_spectrum, &mel_energies);

  mel_energies.ApplyFloor(opts_.energy_floor);
  mel_energies.ApplyLog();  // take the log.

  // Copy energy as first value
  if (opts_.use_energy) {
    (*feature)(0) = signal_log_energy;
  }
}

}  // namespace kaldi

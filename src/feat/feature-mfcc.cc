// feat/feature-mfcc.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek
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


#include "feat/feature-mfcc.h"


namespace kaldi {


// Compute liftering coefficients (scaling on cepstral coeffs)
// coeffs are numbered slightly differently from HTK: the zeroth
// index is C0, which is not affected.
static void ComputeLifterCoeffs(BaseFloat Q, VectorBase<BaseFloat> *coeffs) {
  for (int32 i = 0; i < coeffs->Dim(); i++)
      (*coeffs)(i) = 1.0 + 0.5 * Q * sin (M_PI * i / Q);
}


void MfccComputer::Compute(BaseFloat vtln_warp,
                           VectorBase<BaseFloat> *signal_frame,
                           VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());

  BaseFloat signal_log_energy;
  if (opts_.use_energy)
    signal_log_energy = Log(std::max<BaseFloat>(
        VecVec(*signal_frame, *signal_frame),
        opts_.energy_floor * opts_.frame_opts.WindowSize()));
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  srfft_->Compute(signal_frame->Data(), true);

  // Convert the FFT into a power spectrum.
  ComputePowerSpectrum(signal_frame);
  SubVector<BaseFloat> power_spectrum(*signal_frame, 0,
                                      signal_frame->Dim() / 2 + 1);

  // The energy_floor has the scale for the energy of a single sample, and the
  // FFT has a higher dynamic range (it's not the orthogonal FFT)... the sqrt
  // expression is to correct for that.
  BaseFloat floor = opts_.energy_floor *
                    std::sqrt(BaseFloat(opts_.frame_opts.WindowSize()));
  power_spectrum.ApplyFloor(floor);

  mel_banks.Compute(power_spectrum, &mel_energies_);
  mel_energies_.ApplyLog();

  feature->SetZero();  // in case there were NaNs.
  // feature = dct_matrix_ * mel_energies [which now have log]
  feature->AddMatVec(1.0, dct_matrix_, kNoTrans, mel_energies_, 0.0);
  feature->MulElements(lifter_coeffs_);

  if (opts_.use_energy)
    (*feature)(0) = signal_log_energy;
}

MfccComputer::MfccComputer(const MfccOptions &opts):
    opts_(opts),
    srfft_(new SplitRadixRealFft<BaseFloat>(opts.frame_opts.PaddedWindowSize())),
    mel_energies_(opts.mel_opts.num_bins) {

  int32 num_bins = opts.mel_opts.num_bins;
  if (opts.num_ceps > num_bins)
    KALDI_ERR << "num-ceps cannot be larger than num-mel-bins."
              << " It should be smaller or equal. You provided num-ceps: "
              << opts.num_ceps << "  and num-mel-bins: "
              << num_bins;

  Matrix<BaseFloat> dct_matrix(num_bins, num_bins);
  ComputeDctMatrix(&dct_matrix);
  lifter_coeffs_.Resize(opts.num_ceps);
  ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);


  // Note that we include zeroth dct in either case.  If using the
  // energy we replace this with the energy.  This means a different
  // ordering of features than HTK.
  SubMatrix<BaseFloat> dct_rows(dct_matrix, 0, opts.num_ceps, 0, num_bins);
  dct_matrix_.Resize(opts.num_ceps, num_bins);
  dct_matrix_.CopyFromMat(dct_rows);  // subset of rows.

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

MfccComputer::MfccComputer(const MfccComputer &other):
    opts_(other.opts_), lifter_coeffs_(other.lifter_coeffs_),
    dct_matrix_(other.dct_matrix_),
    mel_banks_(other.mel_banks_),
    srfft_(new SplitRadixRealFft<BaseFloat>(*(other.srfft_))),
    mel_energies_(other.mel_energies_.Dim(), kUndefined) {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
       iter != mel_banks_.end(); ++iter)
    iter->second = new MelBanks(*(iter->second));
}



MfccComputer::~MfccComputer() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;
  delete srfft_;
}

const MelBanks *MfccComputer::GetMelBanks(BaseFloat vtln_warp) {
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



}  // namespace kaldi

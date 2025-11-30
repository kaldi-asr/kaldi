// feat/mel-computations.cc

// Copyright 2009-2011  Phonexia s.r.o.;  Karel Vesely;  Microsoft Corporation

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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <iostream>

#include "feat/feature-functions.h"
#include "feat/feature-window.h"
#include "feat/mel-computations.h"

namespace kaldi {


MelBanks::MelBanks(const MelBanksOptions &opts,
                   const FrameExtractionOptions &frame_opts,
                   BaseFloat vtln_warp_factor):
    htk_mode_(opts.htk_mode) {
  SetConfigs(opts, frame_opts, vtln_warp_factor);

  int32 num_bins = opts.num_bins;
  if (num_bins < 3) KALDI_ERR << "Must have at least 3 mel bins";


  BaseFloat mel_low_freq = MelScale(low_freq_);
  BaseFloat mel_high_freq = MelScale(high_freq_);



  bins_.resize(num_bins);
  center_freqs_.Resize(num_bins);

  for (int32 bin = 0; bin < num_bins; bin++) {
    BaseFloat mel = mel_low_freq +
        (bin + 1) * (mel_high_freq - mel_low_freq) / (num_bins + 1);
    if (vtln_warp_factor != 1.0)
      mel = VtlnWarpMelFreq(vtln_warp_factor, mel);
    center_freqs_(bin) = InverseMelScale(mel);
  }

  if (!opts.modified)
    ComputeBins(opts.htk_mode);
  else
    ComputeModifiedBins();

  if (debug_) {
    for (size_t i = 0; i < bins_.size(); i++) {
      KALDI_LOG << "bin " << i << ", offset = " << bins_[i].first
                << ", vec = " << bins_[i].second;
    }
  }
}

void MelBanks::ComputeBins(bool htk_mode) {
  int32 num_bins = center_freqs_.Dim();
  for (int32 bin = 0; bin < num_bins; bin++) {
    // center_mel is the center frequency (in mel) of this bin, and left_mel and
    // right_mel are those of the bins immediately to the left and right.
    BaseFloat center_mel = MelScale(center_freqs_(bin)),
        left_mel = MelScale(bin == 0 ?
                            low_freq_ : center_freqs_(bin - 1)),
        right_mel = MelScale(bin == num_bins - 1 ?
                             high_freq_ : center_freqs_(bin + 1));
    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    Vector<BaseFloat> this_bin(num_fft_bins_);
    int32 first_index = -1, last_index = -1;
    for (int32 i = 0; i < num_fft_bins_; i++) {
      BaseFloat freq = (fft_bin_width_ * i);  // Center frequency of this fft
                                             // bin.
      BaseFloat mel = MelScale(freq);
      if (mel > left_mel && mel < right_mel) {
        BaseFloat weight;
        if (mel <= center_mel)
          weight = (mel - left_mel) / (center_mel - left_mel);
        else
         weight = (right_mel - mel) / (right_mel - center_mel);
        this_bin(i) = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }
    KALDI_ASSERT(first_index != -1 && last_index >= first_index
                 && "You may have set --num-mel-bins too large.");

    bins_[bin].first = first_index;
    int32 size = last_index + 1 - first_index;
    bins_[bin].second.Resize(size);
    bins_[bin].second.CopyFromVec(this_bin.Range(first_index, size));

    // Replicate a bug in HTK, for testing purposes.
    if (htk_mode && bin == 0 && low_freq_ != 0.0)
      bins_[bin].second(0) = 0.0;
  }
}

/*
  Notes on the shape of the modified bins.

  They are shaped like a cosine function from -pi/2 to pi/2 (unlike the standard
  triangular bins).  We define their diameter as the distance between the
  first and last nonzero value (pi for the canonical function).  We choose
  the diameter as:
       d = sqrt(d1^2 + d2^2)
  (this function may be viewed as a kind of soft-max), where d1 and d2 are
  two different formulas for the diameter that we describe below.

    d1 is a formula that ensures the bins overlap by at least a minimal amount.

   Let bin_diff be the difference in Hz between this bin's center-frequency
   and the next bin's center-frequency, or (if this is the last bin),
   the user-specified `high-freq` which is the top of the range of frequencies
   we cover.  Then:

       d1 = 1.1 * bin_diff

   The formula for d2 is designed to provide a reasonable floor so the bandwidth
   don't get ridiculously narrow as we add more bins, and to approximate what we
   observed the filter diameters to look like when learning filterbanks via DNNs.
   The formula is:

       d2 = 50 + 50 * f / (f + 700)

   which roughly means: start with a diameter of 50Hz, increasing gradually to
   100Hz for bins with center frequency more than about 700Hz.  There is no
   rocket science behind this formula; it was obtained through a combination of
   trying to match the DNN-learned filterbank bandwidths (cite: Pegah's thesis),
   and manual tuning.
 */
void MelBanks::ComputeModifiedBins() {
  int32 num_bins = center_freqs_.Dim();
  for (int32 bin = 0; bin < num_bins; bin++) {
    BaseFloat center_freq = center_freqs_(bin),
        next_center = (bin == num_bins - 1 ?
                       high_freq_ : center_freqs_(bin + 1));

    BaseFloat d1 = (next_center - center_freq) * 1.1,
        d2 = 60.0 + 50.0 * (center_freq / (center_freq + breakpoint_));

    // 'diameter' is in Hz; it represents the distance on the frequency axis
    // between the first and last nonzero points of the raised-cosine window
    // function.  This formula applies our heuristic, described above,
    // to choose the diameter.
    BaseFloat diameter = sqrt(d1 * d1 + d2 * d2);

    // 'freq_scale' is the scaling factor on the frequencies that will ensure
    // that the diameter becomes equal to pi, like the canonical bin function
    // (the cosine from -pi/2 to pi/2).
    BaseFloat freq_scale = M_PI / diameter;

    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    Vector<BaseFloat> this_bin(num_fft_bins_);
    int32 first_index = -1, last_index = -1;

    for (int32 i = 0; i < num_fft_bins_; i++) {
      BaseFloat freq = (fft_bin_width_ * i);  // Center frequency of this fft
                                             // bin.
      BaseFloat normalized_freq = freq_scale * (freq - center_freq);
      if (normalized_freq > -M_PI_2 && normalized_freq < M_PI_2) {
        BaseFloat weight = cos(normalized_freq);
        this_bin(i) = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }
    KALDI_ASSERT(first_index != -1 && last_index >= first_index
                 && "You may have set --num-mel-bins too large.");

    bins_[bin].first = first_index;
    int32 size = last_index + 1 - first_index;
    bins_[bin].second.Resize(size);
    bins_[bin].second.CopyFromVec(this_bin.Range(first_index, size));
  }
}

BaseFloat MelBanks::VtlnWarpFreq(BaseFloat vtln_warp_factor,
                                 BaseFloat freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq and
  /// high_freq inclusive, with the following properties:
  ///  F(low_freq) == low_freq
  ///  F(high_freq) == high_freq
  /// The function is continuous and piecewise linear with two inflection
  ///   points.
  /// The lower inflection point (measured in terms of the unwarped
  ///  frequency) is at frequency l, determined as described below.
  /// The higher inflection point is at a frequency h, determined as
  ///   described below.
  /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
  /// If the higher inflection point (measured in terms of the unwarped
  ///   frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
  ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
  ///   max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
  ///   h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
  ///     = vtln_high_cutoff * min(1, vtln_warp_factor).
  /// If the lower inflection point (measured in terms of the unwarped
  ///   frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
  ///   This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
  ///                       = vtln_low_cutoff * max(1, vtln_warp_factor)


  if (freq < low_freq_ || freq > high_freq_) return freq;  // in case this gets called
  // for out-of-range frequencies, just return the freq.

  BaseFloat l = vtln_low_ * std::max(BaseFloat(1.0), vtln_warp_factor);
  BaseFloat h = vtln_high_ * std::min(BaseFloat(1.0), vtln_warp_factor);
  BaseFloat scale = 1.0 / vtln_warp_factor;
  BaseFloat Fl = scale * l;  // F(l);
  BaseFloat Fh = scale * h;  // F(h);
  KALDI_ASSERT(l > low_freq_ && h < high_freq_);
  // slope of left part of the 3-piece linear function
  BaseFloat scale_left = (Fl - low_freq_) / (l - low_freq_);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  BaseFloat scale_right = (high_freq_ - Fh) / (high_freq_ - h);

  if (freq < l) {
    return low_freq_ + scale_left * (freq - low_freq_);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq_ + scale_right * (freq - high_freq_);
  }
}

BaseFloat MelBanks::VtlnWarpMelFreq(BaseFloat vtln_warp_factor,
                                    BaseFloat mel_freq) {
  return MelScale(VtlnWarpFreq(vtln_warp_factor, InverseMelScale(mel_freq)));
}


// "power_spectrum" contains fft energies.
void MelBanks::Compute(const VectorBase<BaseFloat> &power_spectrum,
                       VectorBase<BaseFloat> *mel_energies_out) const {
  int32 num_bins = bins_.size();
  KALDI_ASSERT(mel_energies_out->Dim() == num_bins);

  for (int32 i = 0; i < num_bins; i++) {
    int32 offset = bins_[i].first;
    const Vector<BaseFloat> &v(bins_[i].second);
    BaseFloat energy = VecVec(v, power_spectrum.Range(offset, v.Dim()));
    // HTK-like flooring- for testing purposes (we prefer dither)
    if (htk_mode_ && energy < 1.0) energy = 1.0;
    (*mel_energies_out)(i) = energy;

    // The following assert was added due to a problem with OpenBlas that
    // we had at one point (it was a bug in that library).  Just to detect
    // it early.
    KALDI_ASSERT(!KALDI_ISNAN((*mel_energies_out)(i)));
  }
}

void MelBanks::SetConfigs(const MelBanksOptions &opts,
                          const FrameExtractionOptions &frame_opts,
                          BaseFloat vtln_warp_factor) {
  BaseFloat sample_freq = frame_opts.samp_freq,
      nyquist = 0.5 * sample_freq;
  int32 window_length_padded = frame_opts.PaddedWindowSize();
  KALDI_ASSERT(window_length_padded % 2 == 0);
  num_fft_bins_ = window_length_padded / 2;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]
  fft_bin_width_ = sample_freq / window_length_padded;

  debug_ = opts.debug_mel;


  low_freq_ = opts.low_freq;
  if (opts.high_freq > 0.0)
    high_freq_ = opts.high_freq;
  else
    high_freq_ = nyquist + opts.high_freq;

  if (low_freq_ < 0.0 || low_freq_ >= nyquist
      || high_freq_ <= 0.0 || high_freq_ > nyquist
      || high_freq_ <= low_freq_)
    KALDI_ERR << "Bad values in options: low-freq " << low_freq_
              << " and high-freq " << high_freq_ << " vs. nyquist "
              << nyquist;

  breakpoint_ = (opts.modified ? 300.0 : 700.0);
  second_breakpoint_ = (opts.modified ? 2000.0 : -1);
  vtln_low_ = opts.vtln_low;
  if (opts.vtln_high > 0.0)
    vtln_high_ = opts.vtln_high;
  else
    vtln_high_ = opts.vtln_high + nyquist;

  if (vtln_warp_factor != 1.0 &&
      (vtln_low_ < 0.0 || vtln_low_ <= low_freq_
       || vtln_low_ >= high_freq_
       || vtln_high_ <= 0.0 || vtln_high_ >= high_freq_
       || vtln_high_ <= vtln_low_))
    KALDI_ERR << "Bad values in options: vtln-low " << vtln_low_
              << " and vtln-high " << vtln_high_ << ", versus "
              << "low-freq " << low_freq_ << " and high-freq "
              << high_freq_;
}


void ComputeLifterCoeffs(BaseFloat Q, VectorBase<BaseFloat> *coeffs) {
  // Compute liftering coefficients (scaling on cepstral coeffs)
  // coeffs are numbered slightly differently from HTK: the zeroth
  // index is C0, which is not affected.
  for (int32 i = 0; i < coeffs->Dim(); i++)
    (*coeffs)(i) = 1.0 + 0.5 * Q * sin (M_PI * i / Q);
}


// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i-1] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the demoninator
BaseFloat Durbin(int n, const BaseFloat *pAC, BaseFloat *pLP, BaseFloat *pTmp) {
  BaseFloat ki;                // reflection coefficient
  int i;
  int j;

  BaseFloat E = pAC[0];

  for (i = 0; i < n; i++) {
    // next reflection coefficient
    ki = pAC[i + 1];
    for (j = 0; j < i; j++)
      ki += pLP[j] * pAC[i - j];
    ki = ki / E;

    // new error
    BaseFloat c = 1 - ki * ki;
    if (c < 1.0e-5) // remove NaNs for constan signal
      c = 1.0e-5;
    E *= c;

    // new LP coefficients
    pTmp[i] = -ki;
    for (j = 0; j < i; j++)
      pTmp[j] = pLP[j] - ki * pLP[i - j - 1];

    for (j = 0; j <= i; j++)
      pLP[j] = pTmp[j];
  }

  return E;
}


void Lpc2Cepstrum(int n, const BaseFloat *pLPC, BaseFloat *pCepst) {
  for (int32 i = 0; i < n; i++) {
    double sum = 0.0;
    int j;
    for (j = 0; j < i; j++) {
      sum += static_cast<BaseFloat>(i - j) * pLPC[j] * pCepst[i - j - 1];
    }
    pCepst[i] = -pLPC[i] - sum / static_cast<BaseFloat>(i + 1);
  }
}

void GetEqualLoudnessVector(const MelBanks &mel_banks,
                            Vector<BaseFloat> *ans) {
  int32 n = mel_banks.NumBins();
  // Central frequency of each mel bin.
  const Vector<BaseFloat> &f0 = mel_banks.GetCenterFreqs();
  ans->Resize(n);
  for (int32 i = 0; i < n; i++) {
    BaseFloat fsq = f0(i) * f0(i);
    BaseFloat fsub = fsq / (fsq + 1.6e5);
    (*ans)(i) = fsub * fsub * ((fsq + 1.44e6) / (fsq + 9.61e6));
  }
}


// Compute LP coefficients from autocorrelation coefficients.
BaseFloat ComputeLpc(const VectorBase<BaseFloat> &autocorr_in,
                     Vector<BaseFloat> *lpc_out) {
  int32 n = autocorr_in.Dim() - 1;
  KALDI_ASSERT(lpc_out->Dim() == n);
  Vector<BaseFloat> tmp(n);
  BaseFloat ans = Durbin(n, autocorr_in.Data(),
                         lpc_out->Data(),
                         tmp.Data());
  if (ans <= 0.0)
    KALDI_WARN << "Zero energy in LPC computation";
  return -Log(1.0 / ans);  // forms the C0 value
}


}  // namespace kaldi

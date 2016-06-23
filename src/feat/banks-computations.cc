// feat/banks-computations.cc

// Copyright 2009-2011  Phonexia s.r.o.;  Karel Vesely;  Microsoft Corporation
//           2016       CereProc Ltd. (author: Blaise Potard)

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

#include "feat/feature-functions.h"
#include "feat/feature-window.h"
#include "feat/banks-computations.h"

namespace kaldi {

FrequencyBanks::FrequencyBanks(const FrequencyBanksOptions &opts,
                               const FrameExtractionOptions &frame_opts,
                               BaseFloat vtln_warp_factor):
  htk_mode_(opts.htk_mode) {
  int32 num_bins = opts.num_bins;
  if (num_bins < 3) KALDI_ERR << "Must have at least 3 bins";
  BaseFloat sample_freq = frame_opts.samp_freq;
  int32 window_length = static_cast<int32>(frame_opts.samp_freq / 1000.0 *
                                           frame_opts.frame_length_ms);
  int32 window_length_padded =
    (frame_opts.round_to_power_of_two ?
     RoundUpToNearestPowerOfTwo(window_length) :
     window_length);
  KALDI_ASSERT(window_length_padded % 2 == 0);
  int32 num_fft_bins = window_length_padded/2;
  BaseFloat nyquist = 0.5 * sample_freq;

  BaseFloat low_freq = opts.low_freq, high_freq;
  if (opts.high_freq > 0.0)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  if (low_freq < 0.0 || low_freq >= nyquist
      || high_freq <= 0.0 || high_freq > nyquist
      || high_freq <= low_freq)
    KALDI_ERR << "Bad values in options: low-freq " << low_freq
              << " and high-freq " << high_freq << " vs. nyquist "
              << nyquist;

  BaseFloat fft_bin_width = sample_freq / window_length_padded;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]

  BaseFloat scaled_low_freq = Scale(low_freq, opts.scale_type);
  BaseFloat scaled_high_freq = Scale(high_freq, opts.scale_type);

  debug_ = opts.debug_banks;

  // divide by num_bins+1 in next line because of end-effects where the bins
  // spread out to the sides.
  BaseFloat scaled_freq_delta = (scaled_high_freq - scaled_low_freq) /
      (num_bins+1);

  BaseFloat vtln_low = opts.vtln_low,
      vtln_high = opts.vtln_high;
  if (vtln_high < 0.0) vtln_high += nyquist;

  if (vtln_warp_factor != 1.0 &&
      (vtln_low < 0.0 || vtln_low <= low_freq
       || vtln_low >= high_freq
       || vtln_high <= 0.0 || vtln_high >= high_freq
       || vtln_high <= vtln_low))
    KALDI_ERR << "Bad values in options: vtln-low " << vtln_low
              << " and vtln-high " << vtln_high << ", versus "
              << "low-freq " << low_freq << " and high-freq "
              << high_freq;

  bins_.resize(num_bins);
  center_freqs_.Resize(num_bins);

  for (int32 bin = 0; bin < num_bins; bin++) {
    BaseFloat left_scaled = scaled_low_freq + bin * scaled_freq_delta,
        center_scaled = scaled_low_freq + (bin + 1) * scaled_freq_delta,
        right_scaled = scaled_low_freq + (bin + 2) * scaled_freq_delta;

    if (vtln_warp_factor != 1.0) {
      left_scaled = VtlnWarpFreq(vtln_low, vtln_high, low_freq, high_freq,
                              vtln_warp_factor, left_scaled, opts.scale_type);
      center_scaled = VtlnWarpFreq(vtln_low, vtln_high, low_freq, high_freq,
                              vtln_warp_factor, center_scaled, opts.scale_type);
      right_scaled = VtlnWarpFreq(vtln_low, vtln_high, low_freq, high_freq,
                              vtln_warp_factor, right_scaled, opts.scale_type);
    }
    center_freqs_(bin) = InverseScale(center_scaled, opts.scale_type);

    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    Vector<BaseFloat> this_bin(num_fft_bins);
    int32 first_index = -1, last_index = -1;
    for (int32 i = 0; i < num_fft_bins; i++) {
      BaseFloat freq = (fft_bin_width * i);  // Center frequency of this fft
                                             // bin.
      BaseFloat scaled = Scale(freq, opts.scale_type);

      // TODO(BP): add support for other type of interpolation apart from
      // triangular, e.g. rectangular and hanning
      if (scaled > left_scaled && scaled < right_scaled) {
        BaseFloat weight;
        if (scaled <= center_scaled)
          weight = (scaled - left_scaled) / (center_scaled - left_scaled);
        else
          weight = (right_scaled - scaled) / (right_scaled - center_scaled);
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
    if (opts.htk_mode && bin == 0 && scaled_low_freq != 0.0)
      bins_[bin].second(0) = 0.0;
  }
  if (debug_) {
    for (size_t i = 0; i < bins_.size(); i++) {
      KALDI_LOG << "bin " << i << ", offset = " << bins_[i].first
                << ", vec = " << bins_[i].second;
    }
  }
}

FrequencyBanks::FrequencyBanks(const FrequencyBanks &other)
    : center_freqs_(other.center_freqs_),
      bins_(other.bins_),
      debug_(other.debug_),
      htk_mode_(other.htk_mode_) { }

BaseFloat FrequencyBanks::VtlnWarpFreqLinear(BaseFloat vtln_low_cutoff,
                                             BaseFloat vtln_high_cutoff,
                                             BaseFloat low_freq,
                                             BaseFloat high_freq,
                                             BaseFloat vtln_warp_factor,
                                             BaseFloat freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq
  /// and high_freq inclusive, with the following properties:
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


  if (freq < low_freq || freq > high_freq) return freq;  // in case this gets
  // called for out-of-range frequencies, just return the freq.

  KALDI_ASSERT(vtln_low_cutoff > low_freq &&
               "be sure to set the --vtln-low option higher than --low-freq");
  KALDI_ASSERT(vtln_high_cutoff < high_freq &&
               "be sure to set the --vtln-high option lower than --high-freq "
               "[or negative]");
  BaseFloat one = 1.0;
  BaseFloat l = vtln_low_cutoff * std::max(one, vtln_warp_factor);
  BaseFloat h = vtln_high_cutoff * std::min(one, vtln_warp_factor);
  BaseFloat scale = 1.0 / vtln_warp_factor;
  BaseFloat Fl = scale * l;  // F(l);
  BaseFloat Fh = scale * h;  // F(h);
  KALDI_ASSERT(l > low_freq && h < high_freq);
  // slope of left part of the 3-piece linear function
  BaseFloat scale_left = (Fl - low_freq) / (l - low_freq);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  BaseFloat scale_right = (high_freq - Fh) / (high_freq - h);

  if (freq < l) {
    return low_freq + scale_left * (freq - low_freq);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq + scale_right * (freq - high_freq);
  }
}

BaseFloat FrequencyBanks::VtlnWarpFreq(BaseFloat vtln_low_cutoff,
                                       BaseFloat vtln_high_cutoff,
                                       BaseFloat low_freq,
                                       BaseFloat high_freq,
                                       BaseFloat vtln_warp_factor,
                                       BaseFloat scaled_freq,
                                       std::string scale_type) {
  return Scale(VtlnWarpFreqLinear(vtln_low_cutoff, vtln_high_cutoff,
                                  low_freq, high_freq,
                                  vtln_warp_factor,
                                  InverseScale(scaled_freq, scale_type)),
               scale_type);
}


// "power_spectrum" contains fft energies.
void FrequencyBanks::Compute(const VectorBase<BaseFloat> &power_spectrum,
                       VectorBase<BaseFloat> *banks_energies_out) const {
  int32 num_bins = bins_.size();
  KALDI_ASSERT(banks_energies_out->Dim() == num_bins);

  for (int32 i = 0; i < num_bins; i++) {
    int32 offset = bins_[i].first;
    const Vector<BaseFloat> &v(bins_[i].second);
    BaseFloat energy = VecVec(v, power_spectrum.Range(offset, v.Dim()));
    // HTK-like flooring- for testing purposes (we prefer dither)
    if (htk_mode_ && energy < 1.0) energy = 1.0;
    (*banks_energies_out)(i) = energy;

    // The following assert was added due to a problem with OpenBlas that
    // we had at one point (it was a bug in that library).  Just to detect
    // it early.
    KALDI_ASSERT(!KALDI_ISNAN((*banks_energies_out)(i)));
  }

  if (debug_) {
    fprintf(stderr, "MEL BANKS:\n");
    for (int32 i = 0; i < num_bins; i++)
      fprintf(stderr, " %f", (*banks_energies_out)(i));
    fprintf(stderr, "\n");
  }
}

}  // namespace kaldi

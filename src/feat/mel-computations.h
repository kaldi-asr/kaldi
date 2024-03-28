// feat/mel-computations.h

// Copyright 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
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

#ifndef KALDI_FEAT_MEL_COMPUTATIONS_H_
#define KALDI_FEAT_MEL_COMPUTATIONS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"


namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

struct FrameExtractionOptions;  // defined in feature-window.h


struct MelBanksOptions {
  int32 num_bins;  // e.g. 25; number of triangular bins
  BaseFloat low_freq;  // e.g. 20; lower frequency cutoff
  BaseFloat high_freq;  // an upper frequency cutoff; 0 -> no cutoff, negative
  // ->added to the Nyquist frequency to get the cutoff.
  BaseFloat vtln_low;  // vtln lower cutoff of warping function.
  BaseFloat vtln_high;  // vtln upper cutoff of warping function: if negative, added
                        // to the Nyquist frequency to get the cutoff.
  bool modified;       // If true, use 'modified' MFCC.
  bool debug_mel;
  // htk_mode is a "hidden" config, it does not show up on command line.
  // Enables more exact compatibility with HTK, for testing purposes.  Affects
  // mel-energy flooring and reproduces a bug in HTK.
  bool htk_mode;
  explicit MelBanksOptions(int num_bins = 25)
      : num_bins(num_bins), low_freq(20), high_freq(0), vtln_low(100),
        vtln_high(-500), modified(false), debug_mel(false), htk_mode(false) {}

  void Register(OptionsItf *opts) {
    opts->Register("num-mel-bins", &num_bins,
                   "Number of triangular mel-frequency bins");
    opts->Register("low-freq", &low_freq,
                   "Low cutoff frequency for mel bins");
    opts->Register("high-freq", &high_freq,
                   "High cutoff frequency for mel bins (if <= 0, offset from Nyquist)");
    opts->Register("vtln-low", &vtln_low,
                   "Low inflection point in piecewise linear VTLN warping function");
    opts->Register("vtln-high", &vtln_high,
                   "High inflection point in piecewise linear VTLN warping function"
                   " (if negative, offset from high-mel-freq");
    opts->Register("modified", &modified,
                   "If true, use a modified form of the Mel scale that gives "
                   "more emphasis to lower frequencies, and use differently "
                   "tuned bin shapes and widths than normal.");
    opts->Register("debug-mel", &debug_mel,
                   "Print out debugging information for mel bin computation");
  }
};


class MelBanks {
 public:

  MelBanks(const MelBanksOptions &opts,
           const FrameExtractionOptions &frame_opts,
           BaseFloat vtln_warp_factor);

  /// Compute Mel energies (note: not log enerties).
  /// At input, "fft_energies" contains the FFT energies (not log).
  void Compute(const VectorBase<BaseFloat> &fft_energies,
               VectorBase<BaseFloat> *mel_energies_out) const;

  int32 NumBins() const { return bins_.size(); }

  // returns vector of central freq of each bin; needed by plp code.
  const Vector<BaseFloat> &GetCenterFreqs() const { return center_freqs_; }

  BaseFloat VtlnWarpFreq(BaseFloat vtln_warp_factor, BaseFloat freq);


  BaseFloat VtlnWarpMelFreq(BaseFloat vtln_warp_factor, BaseFloat mel_freq);

  // Use the default copy constructor
 private:

  // This function checks that the provided options make sense, and also sets
  // configuration variables like breakpoint_ in this class.
  void SetConfigs(const MelBanksOptions &opts,
                  const FrameExtractionOptions &frame_opts,
                  BaseFloat vtln_warp_factor);

  inline BaseFloat InverseMelScale(BaseFloat mel_freq) {
    BaseFloat b1 = breakpoint_, b2 = second_breakpoint_;
    if (b2 > 0.0)  // modified Mel scale
      return b2 * (expf((expf(mel_freq) - b1) / b2) - 1.0);
    else
      return b1 * (expf(mel_freq) - 1.0);
  }

  inline BaseFloat MelScale(BaseFloat freq) {
    BaseFloat b1 = breakpoint_, b2 = second_breakpoint_;
    if (b2 > 0.0) {
      // Modified Mel: linear, till ~b1, then log till ~b2, then log(log)
      return log (b1 + b2 * log(1.0 + freq / b2));
    } else {
      // Mel: linear till ~b1 = 700, then logarithmic.  We ignore the scaling
      // factor as it makes no difference to our application.
      return log(1.0 + freq / b1);
    }
  }

  // This sets up the 'bins_' member, for the regular (not modified)
  // computation.  It assumes center_freqs_ is already set up.
  // 'htk_mode' is expected to be a copy of opts.htk_mode as given to the
  // constructor.
  void ComputeBins(bool htk_mode);

  // This sets up the 'bins_' member, for the modified computaion
  // with cosine-shaped bins that are more tightly
  // computation.  It assumes center_freqs_ is already set up.
  // 'htk_mode' is expected to be a copy of opts.htk_mode as given to the
  // constructor.
  void ComputeModifiedBins();

  // Disallow assignment
  MelBanks &operator = (const MelBanks &other);



  // The following few variables are derived from the configuration
  // options passed in; they are used in converting to and from Mel frequencies,
  // and for other purposes.
  BaseFloat breakpoint_;        // The breakpoint of the Mel scale (700) if we
                                // are using mel scale; otherwise the first
                                // breakpoint in the modified-mel scale,
                                // e.g. 300.  Only relevant if --modified=true
  BaseFloat second_breakpoint_; // The second breakpoint used in the modified
                                // mel scale, e.g. 2000.
                                // Only relevant if --modified=true

  BaseFloat low_freq_;  // opts.low_freq
  BaseFloat high_freq_;  // The same as opts.high_freq if it's >= 0, or
                         // otherwise the Nyquist plus opts.high_freq.
  BaseFloat vtln_low_;  // opts.vtln_low; the lower cutoff for VTLN.
  BaseFloat vtln_high_;  // opts.vtln_high; the upper cutoff for VTLN.

  int32 num_fft_bins_;  // The number of FFT frequency bins (actually, excluding
                        // the one at the Nyquist).  Equal to half the padded
                        // window length.
  BaseFloat fft_bin_width_;  // The frequency separation between successive
                             // FFT bins: equal nyquist / num_fft_bins_.


  // center frequencies of bins (in Hz), numbered from 0 ... num_bins-1.  Needed
  // by GetCenterFreqs().
  Vector<BaseFloat> center_freqs_;

  // the "bins_" vector is a vector, one for each mel bin, of a pair: (the
  // first nonzero fft-bin), (the vector of weights).  The pair of (int32,
  // Vector) is provided for efficiency, to avoid having a larger vector with
  // many zero entries.
  std::vector<std::pair<int32, Vector<BaseFloat> > > bins_;

  bool debug_;
  bool htk_mode_;
};


// Compute liftering coefficients (scaling on cepstral coeffs)
// coeffs are numbered slightly differently from HTK: the zeroth
// index is C0, which is not affected.
void ComputeLifterCoeffs(BaseFloat Q, VectorBase<BaseFloat> *coeffs);


// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i-1] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the denominator
// Returns log energy of residual (I think)
BaseFloat Durbin(int n, const BaseFloat *pAC, BaseFloat *pLP, BaseFloat *pTmp);

// Compute LP coefficients from autocorrelation coefficients.
// Returns log energy of residual (I think)
BaseFloat ComputeLpc(const VectorBase<BaseFloat> &autocorr_in,
                     Vector<BaseFloat> *lpc_out);

void Lpc2Cepstrum(int n, const BaseFloat *pLPC, BaseFloat *pCepst);



void GetEqualLoudnessVector(const MelBanks &mel_banks,
                            Vector<BaseFloat> *ans);

/// @} End of "addtogroup feat"
}  // namespace kaldi

#endif  // KALDI_FEAT_MEL_COMPUTATIONS_H_

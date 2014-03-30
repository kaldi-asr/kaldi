// feat/mel-computations.h

// Copyright 2009-2011  Phonexia s.r.o.;  Microsoft Corporation

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

struct FrameExtractionOptions;  // defined in feature-function.h

struct MelBanksOptions;  // defined in feature-function.h

class MelBanks {
 public:

  static inline BaseFloat InverseMelScale(BaseFloat mel_freq) {
    return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
  }

  static inline BaseFloat MelScale(BaseFloat freq) {
    return 1127.0f * logf (1.0f + freq / 700.0f);
  }

  static BaseFloat VtlnWarpFreq(BaseFloat vtln_low_cutoff,
                                BaseFloat vtln_high_cutoff,  // discontinuities in warp func
                                BaseFloat low_freq,
                                BaseFloat high_freq,  // upper+lower frequency cutoffs in
                                // the mel computation
                                BaseFloat vtln_warp_factor,
                                BaseFloat freq);

  static BaseFloat VtlnWarpMelFreq(BaseFloat vtln_low_cutoff,
                                   BaseFloat vtln_high_cutoff,
                                   BaseFloat low_freq,
                                   BaseFloat high_freq,
                                   BaseFloat vtln_warp_factor,
                                   BaseFloat mel_freq);


  MelBanks(const MelBanksOptions &opts,
           const FrameExtractionOptions &frame_opts,
           BaseFloat vtln_warp_factor);

  /// Compute Mel energies (note: not log enerties).
  /// At input, "fft_energies" contains the FFT energies (not log).
  void Compute(const VectorBase<BaseFloat> &fft_energies,
               Vector<BaseFloat> *mel_energies_out) const;

  int32 NumBins() const { return bins_.size(); }

  // returns vector of central freq of each bin; needed by plp code.
  const Vector<BaseFloat> &GetCenterFreqs() const { return center_freqs_; }

 private:
  // center frequencies of bins, numbered from 0 ... num_bins-1.
  // Needed by GetCenterFreqs().
  Vector<BaseFloat> center_freqs_;

  // the "bins_" vector is a vector, one for each bin, of a pair:
  // (the first nonzero fft-bin), (the vector of weights).
  std::vector<std::pair<int32, Vector<BaseFloat> > > bins_;

  bool debug_;
  bool htk_mode_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(MelBanks);
};


// Compute liftering coefficients (scaling on cepstral coeffs)
// coeffs are numbered slightly differently from HTK: the zeroth
// index is C0, which is not affected.
void ComputeLifterCoeffs(BaseFloat Q, VectorBase<BaseFloat> *coeffs);


// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the demoninator
BaseFloat Durbin(int n, const BaseFloat *pAC, BaseFloat *pLP, BaseFloat *pTmp);

void Lpc2Cepstrum(int n, const BaseFloat *pLPC, BaseFloat *pCepst);

/// @} End of "addtogroup feat"
}  // namespace kaldi

#endif  // KALDI_FEAT_MEL_COMPUTATIONS_H_

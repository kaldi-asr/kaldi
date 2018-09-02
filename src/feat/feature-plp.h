// feat/feature-plp.h

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

#ifndef KALDI_FEAT_FEATURE_PLP_H_
#define KALDI_FEAT_FEATURE_PLP_H_

#include <map>
#include <string>

#include "feat/feature-common.h"
#include "feat/feature-functions.h"
#include "feat/feature-window.h"
#include "feat/mel-computations.h"
#include "itf/options-itf.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{



/// PlpOptions contains basic options for computing PLP features.
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct PlpOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  int32 lpc_order;
  int32 num_ceps;  // num cepstra including zero
  bool use_energy;  // use energy; else C0
  BaseFloat energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing
  BaseFloat compress_factor;
  int32 cepstral_lifter;
  BaseFloat cepstral_scale;

  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of
                    // sqrt(2) on C0 to be the same as HTK.

  PlpOptions() : mel_opts(23),
                 // default number of mel-banks for the PLP computation; this
                 // seems to be common for 16kHz-sampled data. For 8kHz-sampled
                 // data, 15 may be better.
                 lpc_order(12),
                 num_ceps(13),
                 use_energy(true),
                 energy_floor(0.0),
                 raw_energy(true),
                 compress_factor(0.33333),
                 cepstral_lifter(22),
                 cepstral_scale(1.0),
                 htk_compat(false) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("lpc-order", &lpc_order,
                   "Order of LPC analysis in PLP computation");
    opts->Register("num-ceps", &num_ceps,
                   "Number of cepstra in PLP computation (including C0)");
    opts->Register("use-energy", &use_energy,
                   "Use energy (not C0) for zeroth PLP feature");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in PLP computation. "
                   "Only makes a difference if --use-energy=true; only necessary if "
                   "--dither=0.0.  Suggested values: 0.1 or 1.0");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("compress-factor", &compress_factor,
                   "Compression factor in PLP computation");
    opts->Register("cepstral-lifter", &cepstral_lifter,
                   "Constant that controls scaling of PLPs");
    opts->Register("cepstral-scale", &cepstral_scale,
                   "Scaling constant in PLP computation");
    opts->Register("htk-compat", &htk_compat,
                   "If true, put energy or C0 last.  Warning: not sufficient "
                   "to get HTK compatible features (need to change other "
                   "parameters).");
  }
};


/// This is the new-style interface to the PLP computation.
class PlpComputer {
 public:
  typedef PlpOptions Options;
  explicit PlpComputer(const PlpOptions &opts);
  PlpComputer(const PlpComputer &other);

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  int32 Dim() const { return opts_.num_ceps; }

  bool NeedRawLogEnergy() { return opts_.use_energy && opts_.raw_energy; }

  /**
     Function that computes one frame of features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedsRawLogEnergy().
     @param [in] vtln_warp  The VTLN warping factor that the user wants
         to be applied when computing features for this utterance.  Will
         normally be 1.0, meaning no warping is to be done.  The value will
         be ignored for feature types that don't support VLTN, such as
         spectrogram features.
     @param [in] signal_frame  One frame of the signal,
       as extracted using the function ExtractWindow() using the options
       returned by this->GetFrameOptions().  The function will use the
       vector as a workspace, which is why it's a non-const pointer.
     @param [out] feature  Pointer to a vector of size this->Dim(), to which
         the computed feature will be written.
  */
  void Compute(BaseFloat signal_log_energy,
               BaseFloat vtln_warp,
               VectorBase<BaseFloat> *signal_frame,
               VectorBase<BaseFloat> *feature);

  ~PlpComputer();
 private:

  const MelBanks *GetMelBanks(BaseFloat vtln_warp);

  const Vector<BaseFloat> *GetEqualLoudness(BaseFloat vtln_warp);

  PlpOptions opts_;
  Vector<BaseFloat> lifter_coeffs_;
  Matrix<BaseFloat> idft_bases_;
  BaseFloat log_energy_floor_;
  std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
  std::map<BaseFloat, Vector<BaseFloat>* > equal_loudness_;
  SplitRadixRealFft<BaseFloat> *srfft_;

  // temporary vector used inside Compute; size is opts_.mel_opts.num_bins + 2
  Vector<BaseFloat> mel_energies_duplicated_;
  // temporary vector used inside Compute; size is opts_.lpc_order + 1
  Vector<BaseFloat> autocorr_coeffs_;
  // temporary vector used inside Compute; size is opts_.lpc_order
  Vector<BaseFloat> lpc_coeffs_;
  // temporary vector used inside Compute; size is opts_.lpc_order
  Vector<BaseFloat> raw_cepstrum_;

  // Disallow assignment.
  PlpComputer &operator =(const PlpComputer &other);
};

typedef OfflineFeatureTpl<PlpComputer> Plp;

/// @} End of "addtogroup feat"

}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_PLP_H_

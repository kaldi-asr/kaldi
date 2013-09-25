// feat/feature-mfcc.h

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University

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

#ifndef KALDI_FEAT_FEATURE_MFCC_H_
#define KALDI_FEAT_FEATURE_MFCC_H_

#include <map>
#include <string>

#include "feat/feature-functions.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


/// MfccOptions contains basic options for computing MFCC features
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct MfccOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  int32 num_ceps;  // e.g. 13: num cepstral coeffs, counting zero.
  bool use_energy;  // use energy; else C0
  BaseFloat energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing
  BaseFloat cepstral_lifter;  // Scaling factor on cepstra for HTK compatibility.
                              // if 0.0, no liftering is done.
  bool htk_compat;  // if true, put energy/C0 last and introduce a factor of
                    // sqrt(2) on C0 to be the same as HTK.

  MfccOptions() : mel_opts(23),
                  // defaults the #mel-banks to 23 for the MFCC computations.
                  // this seems to be common for 16khz-sampled data,
                  // but for 8khz-sampled data, 15 may be better.
                  num_ceps(13),
                  use_energy(true),
                  energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
                  raw_energy(true),
                  cepstral_lifter(22.0),
                  htk_compat(false) {}

  void Register(OptionsItf *po) {
    frame_opts.Register(po);
    mel_opts.Register(po);
    po->Register("num-ceps", &num_ceps,
                 "Number of cepstra in MFCC computation (including C0)");
    po->Register("use-energy", &use_energy,
                 "Use energy (not C0) in MFCC computation");
    po->Register("energy-floor", &energy_floor,
                 "Floor on energy (absolute, not relative) in MFCC computation");
    po->Register("raw-energy", &raw_energy,
                 "If true, compute energy before preemphasis and windowing");
    po->Register("cepstral-lifter", &cepstral_lifter,
                 "Constant that controls scaling of MFCCs");
    po->Register("htk-compat", &htk_compat,
                 "If true, put energy or C0 last and use a factor of sqrt(2) on "
                 "C0.  Warning: not sufficient to get HTK compatible features "
                 "(need to change other parameters).");
  }
};

class MelBanks;


/// Class for computing MFCC features; see \ref feat_mfcc for more information.
class Mfcc {
 public:
  explicit Mfcc(const MfccOptions &opts);
  ~Mfcc();

  int32 Dim() { return opts_.num_ceps; }

  /// Will throw exception on failure (e.g. if file too short for even one
  /// frame).  The output "wave_remainder" is the last frame or two of the
  /// waveform that it would be necessary to include in the next call to Compute
  /// for the same utterance.  It is not exactly the un-processed part (it may
  /// have been partly processed), it's the start of the next window that we
  /// have not already processed.  Will throw exception on failure (e.g. if file
  /// too short for even one frame).
  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output,
               Vector<BaseFloat> *wave_remainder = NULL);

 private:
  const MelBanks *GetMelBanks(BaseFloat vtln_warp);
  MfccOptions opts_;
  Vector<BaseFloat> lifter_coeffs_;
  Matrix<BaseFloat> dct_matrix_;  // matrix we left-multiply by to perform DCT.
  BaseFloat log_energy_floor_;
  std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
  FeatureWindowFunction feature_window_function_;
  SplitRadixRealFft<BaseFloat> *srfft_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Mfcc);
};


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_MFCC_H_

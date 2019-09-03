// feat/feature-mfcc.h

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
//           2014-2019  Johns Hopkins University (author: Daniel Povey)

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

#include "feat/feature-common.h"
#include "feat/feature-functions.h"
#include "feat/feature-window.h"
#include "feat/mel-computations.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


/// MfccOptions contains basic options for computing MFCC features.
struct MfccOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  int32 num_ceps;  // e.g. 13: num cepstral coeffs, counting zero.
  bool use_energy;  // if true, use energy; else C0
  BaseFloat energy_floor;  // Floor on energy, to avoid log(0.0), which will be
                           // multiplied by sqrt(window-length-in-frames) and
                           // applied per FFT bin. The value of 1.0e-09 is
                           // approximately (1.0/32768.0)^2, like a signal value
                           // of +- 1 in a 16-bit recording.
  // cepstral_lifter controls a scaling factor on the cepstra that helps give
  // all the MFCC coeffs a similar dynamic range by scaling up the
  // higher-frequency coefficients.  It's a rather odd formula involving
  // a sigh.   We don't make it configurable.
  BaseFloat cepstral_lifter;

  MfccOptions() : mel_opts(23),
                  num_ceps(13),
                  use_energy(true),
                  energy_floor(1.0e-09),
                  cepstral_lifter(22.0) { }


  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("num-ceps", &num_ceps,
                   "Number of cepstra in MFCC computation (including C0)");
    opts->Register("use-energy", &use_energy,
                   "Use energy (not C0) in MFCC computation");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) of mel bins etc. "
                   "in MFCC computation. ");
  }
};



// This is the new-style interface to the MFCC computation.
class MfccComputer {
 public:
  typedef MfccOptions Options;
  explicit MfccComputer(const MfccOptions &opts);
  MfccComputer(const MfccComputer &other);

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  int32 Dim() const { return opts_.num_ceps; }

  /**
     Function that computes one frame of features from
     one frame of signal.

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
  void Compute(BaseFloat vtln_warp,
               VectorBase<BaseFloat> *signal_frame,
               VectorBase<BaseFloat> *feature);

  ~MfccComputer();
 private:
  // disallow assignment.
  MfccComputer &operator = (const MfccComputer &in);

 protected:
  const MelBanks *GetMelBanks(BaseFloat vtln_warp);

  MfccOptions opts_;
  Vector<BaseFloat> lifter_coeffs_;
  Matrix<BaseFloat> dct_matrix_;  // matrix we left-multiply by to perform DCT.
  std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
  SplitRadixRealFft<BaseFloat> *srfft_;

  // note: mel_energies_ is specific to the frame we're processing, it's
  // just a temporary workspace.
  Vector<BaseFloat> mel_energies_;
};

typedef OfflineFeatureTpl<MfccComputer> Mfcc;


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_MFCC_H_

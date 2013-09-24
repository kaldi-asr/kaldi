// feat/feature-spectrogram.h

// Copyright 2009-2012  Karel Vesely
// Copyright 2012  Navdeep Jaitly

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

#ifndef KALDI_FEAT_FEATURE_SPECTROGRAM_H_
#define KALDI_FEAT_FEATURE_SPECTROGRAM_H_


#include <string>

#include "feat/feature-functions.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


/// SpectrogramOptions contains basic options for computing SPECTROGRAM features
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct SpectrogramOptions {
  FrameExtractionOptions frame_opts;
  BaseFloat energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing

  SpectrogramOptions() :
    energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
    raw_energy(true) {}

  void Register(OptionsItf *po) {
    frame_opts.Register(po);
    po->Register("energy-floor", &energy_floor,
                 "Floor on energy (absolute, not relative) in Spectrogram computation");
    po->Register("raw-energy", &raw_energy,
                 "If true, compute energy before preemphasis and windowing");
  }
};

/// Class for computing SPECTROGRAM features; see \ref feat_mfcc for more information.
class Spectrogram {
 public:
  explicit Spectrogram(const SpectrogramOptions &opts);
  ~Spectrogram();

  /// Will throw exception on failure (e.g. if file too short for
  /// even one frame).
  void Compute(const VectorBase<BaseFloat> &wave,
               Matrix<BaseFloat> *output,
               Vector<BaseFloat> *wave_remainder = NULL);

 private:
  SpectrogramOptions opts_;
  BaseFloat log_energy_floor_;
  FeatureWindowFunction feature_window_function_;
  SplitRadixRealFft<BaseFloat> *srfft_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Spectrogram);
};


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_SPECTROGRAM_H_

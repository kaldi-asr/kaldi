// ivector/voice-activity-detection.h

// Copyright  2013   Daniel Povey

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


#ifndef KALDI_IVECTOR_VOICE_ACTIVITY_DETECTION_H_
#define KALDI_IVECTOR_VOICE_ACTIVITY_DETECTION_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"

namespace kaldi {

/*
  Note: we may move the location of this file in the future, e.g. to feat/
  This code is geared toward speaker-id applications and is not suitable
  for automatic speech recognition (ASR) because it makes independent
  decisions for each frame without imposing any notion of continuity.
*/
 
struct VadEnergyOptions {
  BaseFloat vad_energy_threshold;
  BaseFloat vad_energy_mean_scale;
  int32 vad_frames_context;
  BaseFloat vad_proportion_threshold;
  
  VadEnergyOptions(): vad_energy_threshold(5.0),
                      vad_energy_mean_scale(0.5),
                      vad_frames_context(0),
                      vad_proportion_threshold(0.6) { }
  void Register(OptionsItf *opts) {
    opts->Register("vad-energy-threshold", &vad_energy_threshold,
                   "Constant term in energy threshold for MFCC0 for VAD (also see "
                   "--vad-energy-mean-scale)");
    opts->Register("vad-energy-mean-scale", &vad_energy_mean_scale,
                   "If this is set to s, to get the actual threshold we "
                   "let m be the mean log-energy of the file, and use "
                   "s*m + vad-energy-threshold");
    opts->Register("vad-frames-context", &vad_frames_context,
                   "Number of frames of context on each side of central frame, "
                   "in window for which energy is monitored");
    opts->Register("vad-proportion-threshold", &vad_proportion_threshold,
                   "Parameter controlling the proportion of frames within "
                   "the window that need to have more energy than the "
                   "threshold");
  }
};


/// Compute voice-activity vector for a file: 1 if we judge the frame as
/// voiced, 0 otherwise.  There are no continuity constraints.
/// This method is a very simple energy-based method which only looks
/// at the first coefficient of "input_features", which is assumed to
/// be a log-energy or something similar.  A cutoff is set-- we use 
/// a formula of the general type: cutoff = 5.0 + 0.5 * (average log-energy
/// in this file), and for each frame the decision is based on the
/// proportion of frames in a context window around the current frame,
/// which are above this cutoff.
void ComputeVadEnergy(const VadEnergyOptions &opts,
                      const MatrixBase<BaseFloat> &input_features,
                      Vector<BaseFloat> *output_voiced);


}  // namespace kaldi



#endif  // KALDI_IVECTOR_VOICE_ACTIVITY_DETECTION_H_

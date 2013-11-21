// feat/pitch-functions-speedup.h

// Copyright     2013  Pegah Ghahremani

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


#ifndef KALDI_FEAT_PITCH_FUNCTIONS_H_
#define KALDI_FEAT_PITCH_FUNCTIONS_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>


#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/mel-computations.h"
//#include "feat/feature-functions.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

struct PitchExtractionOptions {
  FrameExtractionOptions frame_opts;
  double min_f0;          // min f0 to search (Hz)
  double max_f0;          // max f0 to search (Hz)
  double soft_min_f0;     // Minimum f0, applied in soft way, must not exceed min-f0
  double penalty_factor;     // cost factor for FO change
  double double_cost;     // cost of exact FO doubling or halving
  double lowpass_cutoff;  // cutoff frequency for Low pass filter
  double upsample_cutoff; // cutoff frequency we apply for upsampling Nccf
  double resample_freq;   // Integer that determines filter width when upsampling NCCF
  double delta_pitch;     // the pitch tolerance in pruning lags
  double nccf_ballast;    // Increasing this factor reduces NCCF for quiet frames,
                          // helping ensure pitch continuity in unvoiced region
  int32 lowpass_filter_width;       // Integer that determines filter width of lowpass filter
  int32 upsample_filter_width;  // Integer that determines filter width when upsampling NCCF
  
  explicit PitchExtractionOptions() :
    min_f0(50),
    max_f0(550),
    soft_min_f0(10.0),
    penalty_factor(0.1),
    double_cost(1000),
    lowpass_cutoff(1500),
    upsample_cutoff(2000),
    resample_freq(4000),
    delta_pitch(0.01),
    nccf_ballast(0.625),
    lowpass_filter_width(2),
    upsample_filter_width(5) {}
  void Register(ParseOptions *po) {
    frame_opts.Register(po);
    po->Register("min-f0", &min_f0,
                 "min. F0 to search for (Hz)");
    po->Register("max-f0", &max_f0,
                 "max. F0 to search for (Hz)");
    po->Register("soft-min-f0", &soft_min_f0,
                 "Minimum f0, applied in soft way, must not exceed min-f0");
    po->Register("penalty-factor", &penalty_factor,
                 "cost factor for FO change.");
    po->Register("lowpass-cutoff", &lowpass_cutoff,
                 "cuttoff frequency for LowPass filter (Hz) ");
    po->Register("upsample-cutoff", &upsample_cutoff,
                 "cuttoff frequency for upsampling filter (Hz) ");
    po->Register("resample-freq", &resample_freq,
                 "Integer that determines filter width when upsampling NCCF");
    po->Register("delta-pitch", &delta_pitch,
                 "Smallest relative change in pitch that our algorithm measures");
    po->Register("nccf-ballast", &nccf_ballast,
                 "Increasing this factor reduces NCCF for quiet frames");
    po->Register("lowpass-filter-width", &lowpass_filter_width,
                 "Integer that determines filter width of lowpass filter, more gives sharper filter");
    po->Register("upsample-filter-width", &upsample_filter_width,
                 "Integer that determines filter width when upsampling NCCF");
  }
  int32 NccfWindowSize() const {
    return static_cast<int32>(resample_freq * 0.001 * frame_opts.frame_length_ms);
  }

  int32 NccfWindowShift() const {
    return static_cast<int32>(resample_freq * 0.001 * frame_opts.frame_shift_ms);
  }

};

struct PostProcessOption {
  BaseFloat pitch_scale;           // the final pitch scaled with this value
  BaseFloat pov_scale;             // the final pov scaled with this value
  BaseFloat delta_pitch_scale;
  int32 normalization_win_size; // Size of window used for moving window nomalization 
  int32 delta_win_size;    
  int32 nonlin_pov;             // nonlinearity warped function for pov feature      
  bool process_pitch;    
  bool add_delta_pitch;
  explicit PostProcessOption() : 
    pitch_scale(2),
    pov_scale(2),
    delta_pitch_scale(10),
    normalization_win_size(151),
    delta_win_size(5),
    nonlin_pov(1),
    process_pitch(true),
    add_delta_pitch(true) {}

  void Register(ParseOptions *po) {
    po->Register("pitch-scale", &pitch_scale,
                  "Term to scale the final pitch value");
    po->Register("pov-scale", &pov_scale,
                 "Term to scale the final pov value");
    po->Register("delta-pitch-scale", &delta_pitch_scale,
                 "Term to scale the final delta pitch");
    po->Register("normalization-win-size", &normalization_win_size,
                 "size of window used for moving window nomalization");
    po->Register("delta-win-size", &delta_win_size,
                 "size of window for extracting delta pitch");
    po->Register("nonlin-pov", &nonlin_pov,
                 "Controls which nonlinearity we use to warp the NCCF to get a POV measure."
                 "If 1, use (1.001 - nccf)^0.15 - 1; "
                 "if 2, use a longer formula that approximates log(POV / (POV-1)).");
    po->Register("process-pitch", &process_pitch,
                 "Process pitch and pov after extraction and apply nonlinearity or WMWN on tham");
    po->Register("add-delta-pitch", &add_delta_pitch,
                "If true, derivative of log-pitch is added to output features");
  }
  std::vector<BaseFloat> Scale() const { 
    BaseFloat coeffs[] = {-0.2, -0.1 , 0, 0.1 , 0.2};
    std::vector<BaseFloat> scale(coeffs, coeffs + sizeof(coeffs) / sizeof(BaseFloat));
    return scale;
  }
};
/// @} End of "addtogroup feat"

} // namespace kaldi

#endif  // KALDI_FEAT_PITCH_FUNCTIONS_H_

// feat/pitch-functions.h

// Copyright     2013  Pegah Ghahremani
//               2014  IMSL, PKU-HKUST (author: Wei Shi)
//               2014  Yanqing Sun, Junjie Wang,
//                     Daniel Povey, Korbinian Riedhammer

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
#include "itf/online-feature-itf.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

struct PitchExtractionOptions {
  // FrameExtractionOptions frame_opts;
  BaseFloat samp_freq;          // sample frequency in hertz
  BaseFloat frame_shift_ms;     // in milliseconds.
  BaseFloat frame_length_ms;    // in milliseconds.
  BaseFloat preemph_coeff;      // Preemphasis coefficient. [use is deprecated.]
  BaseFloat min_f0;             // min f0 to search (Hz)
  BaseFloat max_f0;             // max f0 to search (Hz)
  BaseFloat soft_min_f0;        // Minimum f0, applied in soft way, must not
                                // exceed min-f0
  BaseFloat penalty_factor;     // cost factor for FO change
  BaseFloat lowpass_cutoff;     // cutoff frequency for Low pass filter
  BaseFloat resample_freq;      // Integer that determines filter width when
                                // upsampling NCCF
  BaseFloat delta_pitch;        // the pitch tolerance in pruning lags
  BaseFloat nccf_ballast;       // Increasing this factor reduces NCCF for
                                // quiet frames, helping ensure pitch
                                // continuity in unvoiced region
  int32 lowpass_filter_width;   // Integer that determines filter width of
                                // lowpass filter
  int32 upsample_filter_width;  // Integer that determines filter width when
                                // upsampling NCCF

  // Below are newer config variables, not present in the original paper,
  // that relate to the online pitch extraction algorithm.
  int32 max_frames_latency;     // The maximum number of frames of latency that
                                // we allow the pitch-processing to introduce,
                                // for online operation (this doesn't relate to
                                // the CPU taken;

  int32 frames_per_chunk;       // Only relevant for the function
                                // ComputeKaldiPitch which is called by
                                // compute-kaldi-pitch-feats.  If nonzero, we
                                // provide the input as chunks of this size.
                                // This affects the energy normalization which
                                // has a small effect on the resulting features,
                                // especially at the beginning of a file.  For
                                // best compatibility with online operation
                                // (e.g. if you plan to train models for the
                                // online-deocding setup), you might want to set
                                // this to a small value, like one frame.

  bool nccf_ballast_online;     // This is a "hidden config" used only for
                                // testing the online pitch extraction.  If
                                // true, we compute the signal root-mean-squared
                                // for the ballast term, only up to the current
                                // frame, rather than the end of the current
                                // chunk of signal. This makes the output
                                // insensitive to the chunking, which is useful
                                // for testing purposes.

  explicit PitchExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      preemph_coeff(0.0),
      min_f0(50),
      max_f0(400),
      soft_min_f0(10.0),
      penalty_factor(0.1),
      lowpass_cutoff(1000),
      resample_freq(4000),
      delta_pitch(0.005),
      nccf_ballast(7000),
      lowpass_filter_width(1),
      upsample_filter_width(5),
      max_frames_latency(20),
      frames_per_chunk(0),
      nccf_ballast_online(false) { }
  void Register(OptionsItf *po,
                bool include_online_opts = false) {
    po->Register("sample-frequency", &samp_freq,
                 "Waveform data sample frequency (must match the waveform "
                 "file, if specified there)");
    po->Register("frame-length", &frame_length_ms, "Frame length in "
                 "milliseconds");
    po->Register("frame-shift", &frame_shift_ms, "Frame shift in milliseconds");
    po->Register("preemphasis-coefficient", &preemph_coeff,
                 "Coefficient for use in signal preemphasis (deprecated)");
    po->Register("min-f0", &min_f0,
                 "min. F0 to search for (Hz)");
    po->Register("max-f0", &max_f0,
                 "max. F0 to search for (Hz)");
    po->Register("soft-min-f0", &soft_min_f0,
                 "Minimum f0, applied in soft way, must not exceed min-f0");
    po->Register("penalty-factor", &penalty_factor,
                 "cost factor for FO change.");
    po->Register("lowpass-cutoff", &lowpass_cutoff,
                 "cutoff frequency for LowPass filter (Hz) ");
    po->Register("resample-frequency", &resample_freq,
                 "Frequency that we down-sample the signal to.  Must be "
                 "more than twice lowpass-cutoff");
    po->Register("delta-pitch", &delta_pitch,
                 "Smallest relative change in pitch that our algorithm "
                 "measures");
    po->Register("nccf-ballast", &nccf_ballast,
                 "Increasing this factor reduces NCCF for quiet frames");
    po->Register("lowpass-filter-width", &lowpass_filter_width,
                 "Integer that determines filter width of "
                 "lowpass filter, more gives sharper filter");
    po->Register("upsample-filter-width", &upsample_filter_width,
                 "Integer that determines filter width when upsampling NCCF");
    po->Register("frames-per-chunk", &frames_per_chunk, "Only relevant for "
                 "offline pitch extraction (e.g. compute-kaldi-pitch-feats), "
                 "you can set it to a small nonzero value, such as 1, for "
                 "better feature compatibility with online decoding (affects "
                 "energy normalization in the algorithm)");
    po->Register("nccf-ballast-online", &nccf_ballast_online,
                 "Compute NCCF ballast using online version of the computation "
                 "(more compatible with online feature extraction");
    if (include_online_opts) {
      po->Register("max-frames-latency", &max_frames_latency, "Maximum number "
                   "of frames of latency that we allow pitch tracking to "
                   "introduce into the feature processing");
    }
  }
  /// Returns the window-size in samples, after resampling.  This is the
  /// "basic window size", not the full window size after extending by max-lag.
  int32 NccfWindowSize() const {
    return static_cast<int32>(resample_freq * 0.001 * frame_length_ms);
  }
  /// Returns the window-shift in samples, after resampling.
  int32 NccfWindowShift() const {
    return static_cast<int32>(resample_freq * 0.001 * frame_shift_ms);
  }
};

struct PostProcessPitchOptions {
  BaseFloat pitch_scale;          // the final pitch scaled with this value
  BaseFloat pov_scale;            // the final pov scaled with this value
  BaseFloat delta_pitch_scale;
  BaseFloat delta_pitch_noise_stddev;  // stddev of noise we add to delta-pitch
  int32 normalization_window_size;     // Size of window used for moving window
                                       // normalization
  int32 delta_window;
  bool process_pitch;
  bool add_delta_pitch;
  bool add_raw_log_pitch;
  bool add_normalized_log_pitch;
  bool add_pov_feature;
  explicit PostProcessPitchOptions() :
    pitch_scale(2.0),
    pov_scale(2.0),
    delta_pitch_scale(10.0),
    delta_pitch_noise_stddev(0.005),
    normalization_window_size(151),
    delta_window(2),
    add_delta_pitch(true),
    add_raw_log_pitch(false),
    add_normalized_log_pitch(true),
    add_pov_feature(true) {}

  void Register(ParseOptions *po) {
    po->Register("pitch-scale", &pitch_scale,
                 "Scaling factor for the final normalized log-pitch value");
    po->Register("pov-scale", &pov_scale,
                 "Scaling factor for final POV (probability of voicing) "
                 "feature");
    po->Register("delta-pitch-scale", &delta_pitch_scale,
                 "Term to scale the final delta log-pitch");
    po->Register("delta-pitch-noise-stddev", &delta_pitch_noise_stddev,
                 "Standard deviation for noise we add to the delta log-pitch "
                 "(before scaling); should be about the same as delta-pitch "
                 "option to pitch creation.  The purpose is to get rid of "
                 "peaks in the delta-pitch caused by discretization of pitch "
                 "values.");
    po->Register("normalization-window-size", &normalization_window_size,
                 "Size of window used for moving window nomalization");
    po->Register("delta-window", &delta_window,
                 "Number of frames on each side of central frame, to use for "
                 "delta window.");
    po->Register("add-pov-feature", &add_pov_feature,
                "If true, the warped NCCF is added to output features");
    po->Register("add-normalized-log-pitch", &add_normalized_log_pitch,
                "If true, the log-pitch with POV-weighted mean subtraction "
                "over 1.5 second window is added to output features");
    po->Register("add-delta-pitch", &add_delta_pitch,
                "If true, time derivative of log-pitch is added to output "
                "features");
    po->Register("add-raw-log-pitch", &add_raw_log_pitch,
                 "If true, log(pitch) is added to output features");
  }
};


// We don't want to expose the pitch-extraction internals here as it's
// quite complex, so we use a private implementation.
class OnlinePitchFeatureImpl;


// Note: to start on a new waveform, just construct a new version
// of this object.
class OnlinePitchFeature: public OnlineBaseFeature {
 public:
  explicit OnlinePitchFeature(const PitchExtractionOptions &opts);

  virtual int32 Dim() const { return 2; }

  virtual int32 NumFramesReady() const;

  virtual bool IsLastFrame(int32 frame) const;

  /// Outputs the two-dimensional feature consisting of (pitch, NCCF).  You
  /// should probably post-process this using class OnlinePostProcessPitch.
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);

  virtual void InputFinished();

  virtual ~OnlinePitchFeature();
 private:
  OnlinePitchFeatureImpl *impl_;
};


/// This online-feature class implements post processing of pitch features.
/// Inputs are original 2 dims (pov, pitch). Original pov -> 2 kinds of povs,
//  pitch -> log_pitch -> 2 kinds of pitch.
class OnlinePostProcessPitch: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return dim_; }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame); }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual ~OnlinePostProcessPitch() {  }

  OnlinePostProcessPitch(const PostProcessPitchOptions &opts,
                         OnlinePitchFeature *src);

  void UpdateFromPitch();
  void ComputePostPitch(const VectorBase<BaseFloat> &nccf_append,
                        const VectorBase<BaseFloat> &log_pitch_append);
 private:

  PostProcessPitchOptions opts_;

  OnlineFeatureInterface *src_;

  int32 dim_;

  Matrix<BaseFloat> features_;

  Vector<BaseFloat> pov_;
  Vector<BaseFloat> pov_feature_;
  Vector<BaseFloat> raw_log_pitch_;

  int32 num_frames_;
  int32 num_pitch_frames_;
};


/// This function extracts (pitch, NCCF) per frame, using the pitch extraction
/// method described in "A Pitch Extraction Algorithm Tuned for Automatic Speech
/// Recognition", Pegah Ghahremani, Bagher BabaAli, Daniel Povey, Korbinian
/// Riedhammer, Jan Trmal and Sanjeev Khudanpur, ICASSP 2014.  The output will
/// have as many rows as there are frames, and two columns corresponding to
/// (NCCF, pitch)
void ComputeKaldiPitch(const PitchExtractionOptions &opts,
                       const VectorBase<BaseFloat> &wave,
                       Matrix<BaseFloat> *output);

/// This function processes the raw (NCCF, pitch) quantities computed by
/// ComputeKaldiPitch, and processes them into features.  By default it will
/// output three-dimensional features, (POV-feature, mean-subtracted-log-pitch,
/// delta-of-raw-pitch), but this is configurable in the options.  The number of
/// rows of "output" will be the number of frames (rows) in "input", and the
/// number of columns will be the number of different types of features
/// requested (by default, 3; 4 is the max).  The four config variables
/// --add-pov-feature, --add-normalized-log-pitch, --add-delta-pitch,
/// --add-raw-log-pitch determine which features we create; by default we create
/// the first three.
void PostProcessPitch(const PostProcessPitchOptions &opts,
                      const MatrixBase<BaseFloat> &input,
                      Matrix<BaseFloat> *output);


/// @} End of "addtogroup feat"
}  // namespace kaldi
#endif  // KALDI_FEAT_PITCH_FUNCTIONS_H_

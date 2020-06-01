// feat/pitch-functions.h

// Copyright     2013  Pegah Ghahremani
//               2014  IMSL, PKU-HKUST (author: Wei Shi)
//               2014  Yanqing Sun, Junjie Wang,
//                     Daniel Povey, Korbinian Riedhammer
//                     Xin Lei

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

#ifndef KALDI_FEAT_PITCH_FUNCTIONS_H_
#define KALDI_FEAT_PITCH_FUNCTIONS_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "feat/mel-computations.h"
#include "itf/online-feature-itf.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"

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

  // The maximum number of frames of latency that we allow the pitch-processing
  // to introduce, for online operation. If you set this to a large value,
  // there would be no inaccuracy from the Viterbi traceback (but it might make
  // you wait to see the pitch). This is not very relevant for the online
  // operation: normalization-right-context is more relevant, you
  // can just leave this value at zero.
  int32 max_frames_latency;

  // Only relevant for the function ComputeKaldiPitch which is called by
  // compute-kaldi-pitch-feats. If nonzero, we provide the input as chunks of
  // this size. This affects the energy normalization which has a small effect
  // on the resulting features, especially at the beginning of a file. For best
  // compatibility with online operation (e.g. if you plan to train models for
  // the online-deocding setup), you might want to set this to a small value,
  // like one frame.
  int32 frames_per_chunk;

  // Only relevant for the function ComputeKaldiPitch which is called by
  // compute-kaldi-pitch-feats, and only relevant if frames_per_chunk is
  // nonzero. If true, it will query the features as soon as they are
  // available, which simulates the first-pass features you would get in online
  // decoding. If false, the features you will get will be the same as those
  // available at the end of the utterance, after InputFinished() has been
  // called: e.g. during lattice rescoring.
  bool simulate_first_pass_online;

  // Only relevant for online operation or when emulating online operation
  // (e.g. when setting frames_per_chunk). This is the frame-index on which we
  // recompute the NCCF (e.g. frame-index 500 = after 5 seconds); if the
  // segment ends before this we do it when the segment ends. We do this by
  // re-computing the signal average energy, which affects the NCCF via the
  // "ballast term", scaling the resampled NCCF by a factor derived from the
  // average change in the "ballast term", and re-doing the backtrace
  // computation. Making this infinity would be the most exact, but would
  // introduce unwanted latency at the end of long utterances, for little
  // benefit.
  int32 recompute_frame;

  // This is a "hidden config" used only for testing the online pitch
  // extraction. If true, we compute the signal root-mean-squared for the
  // ballast term, only up to the current frame, rather than the end of the
  // current chunk of signal. This makes the output insensitive to the
  // chunking, which is useful for testing purposes.
  bool nccf_ballast_online;
  bool snip_edges;
  PitchExtractionOptions():
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
      max_frames_latency(0),
      frames_per_chunk(0),
      simulate_first_pass_online(false),
      recompute_frame(500),
      nccf_ballast_online(false),
      snip_edges(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("sample-frequency", &samp_freq,
                   "Waveform data sample frequency (must match the waveform "
                   "file, if specified there)");
    opts->Register("frame-length", &frame_length_ms, "Frame length in "
                   "milliseconds");
    opts->Register("frame-shift", &frame_shift_ms, "Frame shift in "
                   "milliseconds");
    opts->Register("preemphasis-coefficient", &preemph_coeff,
                   "Coefficient for use in signal preemphasis (deprecated)");
    opts->Register("min-f0", &min_f0,
                   "min. F0 to search for (Hz)");
    opts->Register("max-f0", &max_f0,
                   "max. F0 to search for (Hz)");
    opts->Register("soft-min-f0", &soft_min_f0,
                   "Minimum f0, applied in soft way, must not exceed min-f0");
    opts->Register("penalty-factor", &penalty_factor,
                   "cost factor for FO change.");
    opts->Register("lowpass-cutoff", &lowpass_cutoff,
                   "cutoff frequency for LowPass filter (Hz) ");
    opts->Register("resample-frequency", &resample_freq,
                   "Frequency that we down-sample the signal to.  Must be "
                   "more than twice lowpass-cutoff");
    opts->Register("delta-pitch", &delta_pitch,
                   "Smallest relative change in pitch that our algorithm "
                   "measures");
    opts->Register("nccf-ballast", &nccf_ballast,
                   "Increasing this factor reduces NCCF for quiet frames");
    opts->Register("nccf-ballast-online", &nccf_ballast_online,
                   "This is useful mainly for debug; it affects how the NCCF "
                   "ballast is computed.");
    opts->Register("lowpass-filter-width", &lowpass_filter_width,
                   "Integer that determines filter width of "
                   "lowpass filter, more gives sharper filter");
    opts->Register("upsample-filter-width", &upsample_filter_width,
                   "Integer that determines filter width when upsampling NCCF");
    opts->Register("frames-per-chunk", &frames_per_chunk, "Only relevant for "
                   "offline pitch extraction (e.g. compute-kaldi-pitch-feats), "
                   "you can set it to a small nonzero value, such as 10, for "
                   "better feature compatibility with online decoding (affects "
                   "energy normalization in the algorithm)");
    opts->Register("simulate-first-pass-online", &simulate_first_pass_online,
                   "If true, compute-kaldi-pitch-feats will output features "
                   "that correspond to what an online decoder would see in the "
                   "first pass of decoding-- not the final version of the "
                   "features, which is the default.  Relevant if "
                   "--frames-per-chunk > 0");
    opts->Register("recompute-frame", &recompute_frame, "Only relevant for "
                   "online pitch extraction, or for compatibility with online "
                   "pitch extraction.  A non-critical parameter; the frame at "
                   "which we recompute some of the forward pointers, after "
                   "revising our estimate of the signal energy.  Relevant if"
                   "--frames-per-chunk > 0");
    opts->Register("max-frames-latency", &max_frames_latency, "Maximum number "
                   "of frames of latency that we allow pitch tracking to "
                   "introduce into the feature processing (affects output only "
                   "if --frames-per-chunk > 0 and "
                   "--simulate-first-pass-online=true");
    opts->Register("snip-edges", &snip_edges, "If this is set to false, the "
                   "incomplete frames near the ending edge won't be snipped, "
                   "so that the number of frames is the file size divided by "
                   "the frame-shift. This makes different types of features "
                   "give the same number of frames.");
  }
  /// Returns the window-size in samples, after resampling.  This is the
  /// "basic window size", not the full window size after extending by max-lag.
  // Because of floating point representation, it is more reliable to divide
  // by 1000 instead of multiplying by 0.001, but it is a bit slower.
  int32 NccfWindowSize() const {
    return static_cast<int32>(resample_freq * frame_length_ms / 1000.0);
  }
  /// Returns the window-shift in samples, after resampling.
  int32 NccfWindowShift() const {
    return static_cast<int32>(resample_freq * frame_shift_ms / 1000.0);
  }
};

struct ProcessPitchOptions {
  BaseFloat pitch_scale;  // the final normalized-log-pitch feature is scaled
                          // with this value
  BaseFloat pov_scale;    // the final POV feature is scaled with this value
  BaseFloat pov_offset;   // An offset that can be added to the final POV
                          // feature (useful for online-decoding, where we don't
                          // do CMN to the pitch-derived features.

  BaseFloat delta_pitch_scale;
  BaseFloat delta_pitch_noise_stddev;  // stddev of noise we add to delta-pitch
  int32 normalization_left_context;    // left-context used for sliding-window
                                       // normalization
  int32 normalization_right_context;   // this should be reduced in online
                                       // decoding to reduce latency

  int32 delta_window;
  int32 delay;

  bool add_pov_feature;
  bool add_normalized_log_pitch;
  bool add_delta_pitch;
  bool add_raw_log_pitch;

  ProcessPitchOptions() :
      pitch_scale(2.0),
      pov_scale(2.0),
      pov_offset(0.0),
      delta_pitch_scale(10.0),
      delta_pitch_noise_stddev(0.005),
      normalization_left_context(75),
      normalization_right_context(75),
      delta_window(2),
      delay(0),
      add_pov_feature(true),
      add_normalized_log_pitch(true),
      add_delta_pitch(true),
      add_raw_log_pitch(false) { }


  void Register(OptionsItf *opts) {
    opts->Register("pitch-scale", &pitch_scale,
                   "Scaling factor for the final normalized log-pitch value");
    opts->Register("pov-scale", &pov_scale,
                   "Scaling factor for final POV (probability of voicing) "
                   "feature");
    opts->Register("pov-offset", &pov_offset,
                   "This can be used to add an offset to the POV feature. "
                   "Intended for use in online decoding as a substitute for "
                   " CMN.");
    opts->Register("delta-pitch-scale", &delta_pitch_scale,
                   "Term to scale the final delta log-pitch feature");
    opts->Register("delta-pitch-noise-stddev", &delta_pitch_noise_stddev,
                   "Standard deviation for noise we add to the delta log-pitch "
                   "(before scaling); should be about the same as delta-pitch "
                   "option to pitch creation.  The purpose is to get rid of "
                   "peaks in the delta-pitch caused by discretization of pitch "
                   "values.");
    opts->Register("normalization-left-context", &normalization_left_context,
                   "Left-context (in frames) for moving window normalization");
    opts->Register("normalization-right-context", &normalization_right_context,
                   "Right-context (in frames) for moving window normalization");
    opts->Register("delta-window", &delta_window,
                   "Number of frames on each side of central frame, to use for "
                   "delta window.");
    opts->Register("delay", &delay,
                   "Number of frames by which the pitch information is "
                   "delayed.");
    opts->Register("add-pov-feature", &add_pov_feature,
                   "If true, the warped NCCF is added to output features");
    opts->Register("add-normalized-log-pitch", &add_normalized_log_pitch,
                   "If true, the log-pitch with POV-weighted mean subtraction "
                   "over 1.5 second window is added to output features");
    opts->Register("add-delta-pitch", &add_delta_pitch,
                   "If true, time derivative of log-pitch is added to output "
                   "features");
    opts->Register("add-raw-log-pitch", &add_raw_log_pitch,
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

  virtual int32 Dim() const { return 2; /* (NCCF, pitch) */ }

  virtual int32 NumFramesReady() const;

  virtual BaseFloat FrameShiftInSeconds() const;

  virtual bool IsLastFrame(int32 frame) const;

  /// Outputs the two-dimensional feature consisting of (pitch, NCCF).  You
  /// should probably post-process this using class OnlineProcessPitch.
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);

  virtual void InputFinished();

  virtual ~OnlinePitchFeature();

 private:
  OnlinePitchFeatureImpl *impl_;
};


/// This online-feature class implements post processing of pitch features.
/// Inputs are original 2 dims (nccf, pitch).  It can produce various
/// kinds of outputs, using the default options it will be (pov-feature,
/// normalized-log-pitch, delta-log-pitch).
class OnlineProcessPitch: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return dim_; }

  virtual bool IsLastFrame(int32 frame) const {
    if (frame <= -1)
      return src_->IsLastFrame(-1);
    else if (frame < opts_.delay)
      return src_->IsLastFrame(-1) == true ? false : src_->IsLastFrame(0);
    else
      return src_->IsLastFrame(frame - opts_.delay);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual ~OnlineProcessPitch() {  }

  // Does not take ownership of "src".
  OnlineProcessPitch(const ProcessPitchOptions &opts,
                     OnlineFeatureInterface *src);

 private:
  enum { kRawFeatureDim = 2};  // anonymous enum to define a constant.
                               // kRawFeatureDim defines the dimension
                               // of the input: (nccf, pitch)

  ProcessPitchOptions opts_;
  OnlineFeatureInterface *src_;
  int32 dim_;  // Output feature dimension, set in initializer.

  struct NormalizationStats {
    int32 cur_num_frames;      // value of src_->NumFramesReady() when
                               // "mean_pitch" was set.
    bool input_finished;       // true if input data was finished when
                               // "mean_pitch" was computed.
    double sum_pov;            // sum of pov over relevant range
    double sum_log_pitch_pov;  // sum of log(pitch) * pov over relevant range

    NormalizationStats(): cur_num_frames(-1), input_finished(false),
                          sum_pov(0.0), sum_log_pitch_pov(0.0) { }
  };

  std::vector<BaseFloat> delta_feature_noise_;

  std::vector<NormalizationStats> normalization_stats_;

  /// Computes and returns the POV feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetPovFeature(int32 frame) const;

  /// Computes and returns the delta-log-pitch feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetDeltaPitchFeature(int32 frame);

  /// Computes and returns the raw log-pitch feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetRawLogPitchFeature(int32 frame) const;

  /// Computes and returns the mean-subtracted log-pitch feature for this frame.
  /// Called from GetFrame().
  inline BaseFloat GetNormalizedLogPitchFeature(int32 frame);

  /// Computes the normalization window sizes.
  inline void GetNormalizationWindow(int32 frame,
                                     int32 src_frames_ready,
                                     int32 *window_begin,
                                     int32 *window_end) const;

  /// Makes sure the entry in normalization_stats_ for this frame is up to date;
  /// called from GetNormalizedLogPitchFeature.
  inline void UpdateNormalizationStats(int32 frame);
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
void ProcessPitch(const ProcessPitchOptions &opts,
                  const MatrixBase<BaseFloat> &input,
                  Matrix<BaseFloat> *output);

/// This function combines ComputeKaldiPitch and ProcessPitch.  The reason
/// why we need a separate function to do this is in order to be able to
/// accurately simulate the online pitch-processing, for testing and for
/// training models matched to the "first-pass" features.  It is sensitive to
/// the variables in pitch_opts that relate to online processing,
/// i.e. max_frames_latency, frames_per_chunk, simulate_first_pass_online,
/// recompute_frame.
void ComputeAndProcessKaldiPitch(const PitchExtractionOptions &pitch_opts,
                                 const ProcessPitchOptions &process_opts,
                                 const VectorBase<BaseFloat> &wave,
                                 Matrix<BaseFloat> *output);


/// @} End of "addtogroup feat"
}  // namespace kaldi
#endif  // KALDI_FEAT_PITCH_FUNCTIONS_H_

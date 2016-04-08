// feat/feature-window.h

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
//           2014-2016  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FEAT_FEATURE_WINDOW_H_
#define KALDI_FEAT_FEATURE_WINDOW_H_

#include <map>
#include <string>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

struct FrameExtractionOptions {
  BaseFloat samp_freq;
  BaseFloat frame_shift_ms;  // in milliseconds.
  BaseFloat frame_length_ms;  // in milliseconds.
  BaseFloat dither;  // Amount of dithering, 0.0 means no dither.
  BaseFloat preemph_coeff;  // Preemphasis coefficient.
  bool remove_dc_offset;  // Subtract mean of wave before FFT.
  std::string window_type;  // e.g. Hamming window
  bool round_to_power_of_two;
  bool snip_edges;
  // Maybe "hamming", "rectangular", "povey", "hanning"
  // "povey" is a window I made to be similar to Hamming but to go to zero at the
  // edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
  // I just don't think the Hamming window makes sense as a windowing function.
  FrameExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      dither(1.0),
      preemph_coeff(0.97),
      remove_dc_offset(true),
      window_type("povey"),
      round_to_power_of_two(true),
      snip_edges(true){ }

  void Register(OptionsItf *opts) {
    opts->Register("sample-frequency", &samp_freq,
                   "Waveform data sample frequency (must match the waveform file, "
                   "if specified there)");
    opts->Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
    opts->Register("frame-shift", &frame_shift_ms, "Frame shift in milliseconds");
    opts->Register("preemphasis-coefficient", &preemph_coeff,
                   "Coefficient for use in signal preemphasis");
    opts->Register("remove-dc-offset", &remove_dc_offset,
                   "Subtract mean from waveform on each frame");
    opts->Register("dither", &dither, "Dithering constant (0.0 means no dither)");
    opts->Register("window-type", &window_type, "Type of window "
                   "(\"hamming\"|\"hanning\"|\"povey\"|\"rectangular\")");
    opts->Register("round-to-power-of-two", &round_to_power_of_two,
                   "If true, round window size to power of two.");
    opts->Register("snip-edges", &snip_edges,
                   "If true, end effects will be handled by outputting only frames that "
                   "completely fit in the file, and the number of frames depends on the "
                   "frame-length.  If false, the number of frames depends only on the "
                   "frame-shift, and we reflect the data at the ends.");
  }
  int32 WindowShift() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_shift_ms);
  }
  int32 WindowSize() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_length_ms);
  }
  int32 PaddedWindowSize() const {
    return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) :
                                    WindowSize());
  }
};


struct FeatureWindowFunction {
  FeatureWindowFunction() {}
  explicit FeatureWindowFunction(const FrameExtractionOptions &opts);
  Vector<BaseFloat> window;
};

int32 NumFrames(int32 wave_length,
                const FrameExtractionOptions &opts);


void Dither(VectorBase<BaseFloat> *waveform, BaseFloat dither_value);

void Preemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff);


// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size. If log_energy_pre_window != NULL, outputs the log of the
// sum-of-squared samples before preemphasis and windowing
void ExtractWindow(const VectorBase<BaseFloat> &wave,
                   int32 f,  // with 0 <= f < NumFrames(wave.Dim(), opts)
                   const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   Vector<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window = NULL);

// ExtractWaveformRemainder is useful if the waveform is coming in segments.
// It extracts the bit of the waveform at the end of this block that you
// would have to append the next bit of waveform to, if you wanted to have
// the same effect as everything being in one big block.
void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder);


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_WINDOW_H_

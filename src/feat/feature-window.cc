// feat/feature-window.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//           2013-2016  Johns Hopkins University (author: Daniel Povey)
//                2014  IMSL, PKU-HKUST (author: Wei Shi)

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


#include "feat/feature-window.h"
#include "matrix/matrix-functions.h"


namespace kaldi {

int32 NumFrames(int32 nsamp,
                const FrameExtractionOptions &opts) {
  int32 frame_shift = opts.WindowShift();
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  if (opts.snip_edges) {
    if (static_cast<int32>(nsamp) < frame_length)
      return 0;
    else
      return (1 + ((nsamp - frame_length) / frame_shift));
      // view the expression above as: nsamp-frame_length is how much room we
      // have to shift the frame within the waveform; frame_shift is how much
      // we shift it each time and the ratio is how many times we can shift
      // it (integer arithmetic rounds down).
  } else {
    return (int32)(nsamp * 1.0f / frame_shift + 0.5f);
    // if --snip-edges=false, the number of frames would be determined by
    // rounding the (file-length / frame-shift) to the nearest integer
  }
}


void Dither(VectorBase<BaseFloat> *waveform, BaseFloat dither_value) {
  for (int32 i = 0; i < waveform->Dim(); i++)
    (*waveform)(i) += RandGauss() * dither_value;
}


void Preemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  KALDI_ASSERT(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
  for (int32 i = waveform->Dim()-1; i > 0; i--)
    (*waveform)(i) -= preemph_coeff * (*waveform)(i-1);
  (*waveform)(0) -= preemph_coeff * (*waveform)(0);
}

FeatureWindowFunction::FeatureWindowFunction(const FrameExtractionOptions &opts) {
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(frame_length > 0);
  window.Resize(frame_length);
  for (int32 i = 0; i < frame_length; i++) {
    BaseFloat i_fl = static_cast<BaseFloat>(i);
    if (opts.window_type == "hanning") {
      window(i) = 0.5  - 0.5*cos(M_2PI * i_fl / (frame_length-1));
    } else if (opts.window_type == "hamming") {
      window(i) = 0.54 - 0.46*cos(M_2PI * i_fl / (frame_length-1));
    } else if (opts.window_type == "povey") {  // like hamming but goes to zero at edges.
      window(i) = pow(0.5 - 0.5*cos(M_2PI * i_fl / (frame_length-1)), 0.85);
    } else if (opts.window_type == "rectangular") {
      window(i) = 1.0;
    } else {
      KALDI_ERR << "Invalid window type " << opts.window_type;
    }
  }
}

// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size.  It does mean subtraction, pre-emphasis and dithering as
// requested.
void ExtractWindow(const VectorBase<BaseFloat> &wave,
                   int32 f,  // with 0 <= f < NumFrames(feats, opts)
                   const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   Vector<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window) {
  int32 frame_shift = opts.WindowShift();
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(window_function.window.Dim() == frame_length);
  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);

  Vector<BaseFloat> wave_part(frame_length);
  if (opts.snip_edges) {
    int32 start = frame_shift*f, end = start + frame_length;
    KALDI_ASSERT(start >= 0 && end <= wave.Dim());
    wave_part.CopyFromVec(wave.Range(start, frame_length));
  } else {
    // If opts.snip_edges = false, we allow the frames to go slightly over the
    // edges of the file; we'll extend the data by reflection.
    int32 mid = frame_shift * (f + 0.5),
        begin = mid - frame_length / 2,
        end = begin + frame_length,
        begin_limited = std::max<int32>(0, begin),
        end_limited = std::min(end, wave.Dim()),
        length_limited = end_limited - begin_limited;

    // Copy the main part.  Usually this will be the entire window.
    wave_part.Range(begin_limited - begin, length_limited).
        CopyFromVec(wave.Range(begin_limited, length_limited));

    // Deal with any end effects by reflection, if needed.  This code will
    // rarely be reached, so we don't concern ourselves with efficiency.
    for (int32 f = begin; f < 0; f++) {
      int32 reflected_f = -f;
      // The next statement will only have an effect in the case of files
      // shorter than a single frame, it's to avoid a crash in those cases.
      reflected_f = reflected_f % wave.Dim();
      wave_part(f - begin) = wave(reflected_f);
    }
    for (int32 f = wave.Dim(); f < end; f++) {
      int32 distance_to_end = f - wave.Dim();
      // The next statement will only have an effect in the case of files
      // shorter than a single frame, it's to avoid a crash in those cases.
      distance_to_end = distance_to_end % wave.Dim();
      int32 reflected_f = wave.Dim() - 1 - distance_to_end;
      wave_part(f - begin) = wave(reflected_f);
    }
  }
  KALDI_ASSERT(window != NULL);
  int32 frame_length_padded = opts.PaddedWindowSize();

  if (window->Dim() != frame_length_padded)
    window->Resize(frame_length_padded);

  SubVector<BaseFloat> window_part(*window, 0, frame_length);
  window_part.CopyFromVec(wave_part);

  if (opts.dither != 0.0) Dither(&window_part, opts.dither);

  if (opts.remove_dc_offset)
    window_part.Add(-window_part.Sum() / frame_length);

  if (log_energy_pre_window != NULL) {
    BaseFloat energy = std::max(VecVec(window_part, window_part),
                                std::numeric_limits<float>::min());
    *log_energy_pre_window = Log(energy);
  }

  if (opts.preemph_coeff != 0.0)
    Preemphasize(&window_part, opts.preemph_coeff);

  window_part.MulElements(window_function.window);

  if (frame_length != frame_length_padded)
    SubVector<BaseFloat>(*window, frame_length,
                         frame_length_padded-frame_length).SetZero();
}

void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder) {
  int32 frame_shift = opts.WindowShift();
  int32 num_frames = NumFrames(wave.Dim(), opts);
  // offset is the amount at the start that has been extracted.
  int32 offset = num_frames * frame_shift;
  KALDI_ASSERT(wave_remainder != NULL);
  int32 remaining_len = wave.Dim() - offset;
  wave_remainder->Resize(remaining_len);
  KALDI_ASSERT(remaining_len >= 0);
  if (remaining_len > 0)
    wave_remainder->CopyFromVec(SubVector<BaseFloat>(wave, offset, remaining_len));
}


}  // namespace kaldi

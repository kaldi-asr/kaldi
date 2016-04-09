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


int64 FirstSampleOfFrame(int32 frame,
                         const FrameExtractionOptions &opts) {
  int64 frame_shift = opts.WindowShift();
  if (opts.snip_edges) {
    return frame * frame_shift;
  } else {
    int64 midpoint_of_frame = frame_shift * frame  +  frame_shift / 2,
        beginning_of_frame = midpoint_of_frame  -  opts.WindowSize() / 2;
    return beginning_of_frame;
  }
}

int32 NumFrames(int64 num_samples,
                const FrameExtractionOptions &opts,
                bool flush) {
  int64 frame_shift = opts.WindowShift();
  int64 frame_length = opts.WindowSize();
  if (opts.snip_edges) {
    // with --snip-edges=true (the default), we use a HTK-like approach to
    // determining the number of frames-- all frames have to fit completely into
    // the waveform, and the first frame begins at sample zero.
    if (num_samples < frame_length)
      return 0;
    else
      return (1 + ((num_samples - frame_length) / frame_shift));
    // You can understand the expression above as follows: 'num_samples -
    // frame_length' is how much room we have to shift the frame within the
    // waveform; 'frame_shift' is how much we shift it each time; and the ratio
    // is how many times we can shift it (integer arithmetic rounds down).
  } else {
    // if --snip-edges=false, the number of frames is determined by rounding the
    // (file-length / frame-shift) to the nearest integer.  The point of this
    // formula is to make the number of frames an obvious and predictable
    // function of the frame shift and signal length, which makes many
    // segmentation-related questions simpler.
    //
    // Because integer division in C++ rounds toward zero, we add (half the
    // frame-shift minus epsilon) before dividing, to have the effect of
    // rounding towards the closest integer.
    int32 num_frames = (num_samples + (frame_shift / 2)) / frame_shift;

    if (flush)
      return num_frames;

    // note: 'end' always means the last plus one, i.e. one past the last.
    int64 end_sample_of_last_frame = FirstSampleOfFrame(num_frames - 1, opts)
        + frame_length;

    // the following code is optimized more for clarity than efficiency.
    // If flush == false, we can't output frames that extend past the end
    // of the signal.
    while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
      num_frames--;
      end_sample_of_last_frame -= frame_shift;
    }
    return num_frames;
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

void ProcessWindow(const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   VectorBase<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window) {
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(window->Dim() == frame_length);

  if (opts.dither != 0.0)
    Dither(window, opts.dither);

  if (opts.remove_dc_offset)
    window->Add(-window->Sum() / frame_length);

  if (log_energy_pre_window != NULL) {
    BaseFloat energy = std::max(VecVec(*window, *window),
                                std::numeric_limits<float>::epsilon());
    *log_energy_pre_window = Log(energy);
  }

  if (opts.preemph_coeff != 0.0)
    Preemphasize(window, opts.preemph_coeff);

  window->MulElements(window_function.window);
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

  ProcessWindow(opts, window_function, &window_part, log_energy_pre_window);

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

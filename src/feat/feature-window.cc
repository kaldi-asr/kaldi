// feat/feature-window.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//           2013-2019  Johns Hopkins University (author: Daniel Povey)
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
  int64 midpoint_of_frame = frame_shift * frame  +  frame_shift / 2,
      beginning_of_frame = midpoint_of_frame  -  opts.WindowSize() / 2;
  return beginning_of_frame;
}

int32 NumFrames(int64 num_samples,
                const FrameExtractionOptions &opts,
                bool flush) {
  int64 frame_shift = opts.WindowShift();
  int64 frame_length = opts.WindowSize();

  // The number of frames is determined by rounding the
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


void InitFeatureWindowFunction(const FrameExtractionOptions &opts,
                               Vector<BaseFloat> *window_function) {
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(frame_length > 0);
  window_function->Resize(frame_length);
  double a = M_2PI / (frame_length-1);
  for (int32 i = 0; i < frame_length; i++) {
    double i_fl = static_cast<double>(i);
    if (opts.window_type == "hanning") {
      (*window_function)(i) = 0.5  - 0.5*cos(a * i_fl);
    } else if (opts.window_type == "hamming") {
      (*window_function)(i) = 0.54 - 0.46*cos(a * i_fl);
    } else if (opts.window_type == "povey") {  // like hamming but goes to zero at edges.
      (*window_function)(i) = pow(0.5 - 0.5*cos(a * i_fl), 0.85);
    } else if (opts.window_type == "rectangular") {
      (*window_function)(i) = 1.0;
    } else if (opts.window_type == "blackman") {
      (*window_function)(i) = opts.blackman_coeff - 0.5*cos(a * i_fl) +
        (0.5 - opts.blackman_coeff) * cos(2 * a * i_fl);
    } else {
      KALDI_ERR << "Invalid window type " << opts.window_type;
    }
  }
}

void ProcessWindow(const FrameExtractionOptions &opts,
                   const VectorBase<BaseFloat> &window_function,
                   VectorBase<BaseFloat> *window) {
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(window->Dim() == frame_length);


  /*  This was formerly enabled by the --remove-dc-offset option.  Right now that
      option is mandatory.  */
  window->Add(-window->Sum() / frame_length);

  window->MulElements(window_function);
}


// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size.  It does mean subtraction if requested.
void ExtractWindow(int64 sample_offset,
                   const VectorBase<BaseFloat> &wave,
                   int32 f,  // with 0 <= f < NumFrames(feats, opts)
                   const FrameExtractionOptions &opts,
                   const VectorBase<BaseFloat> &window_function,
                   Vector<BaseFloat> *window) {
  KALDI_ASSERT(sample_offset >= 0 && wave.Dim() != 0);
  int32 frame_length = opts.WindowSize(),
      frame_length_padded = opts.PaddedWindowSize();
  int64 start_sample = FirstSampleOfFrame(f, opts);

  KALDI_ASSERT(sample_offset == 0 || start_sample >= sample_offset);

  if (window->Dim() != frame_length_padded)
    window->Resize(frame_length_padded, kUndefined);

  // wave_start and wave_end are start and end indexes into 'wave', for the
  // piece of wave that we're trying to extract.
  int32 wave_start = int32(start_sample - sample_offset),
      wave_end = wave_start + frame_length;
  if (wave_start >= 0 && wave_end <= wave.Dim()) {
    // the normal case-- no edge effects to consider.
    window->Range(0, frame_length).CopyFromVec(
        wave.Range(wave_start, frame_length));
  } else {
    // Deal with any end effects by reflection, if needed.  This code will only
    // be reached for about two frames per utterance, so we don't concern
    // ourselves excessively with efficiency.
    int32 wave_dim = wave.Dim();
    for (int32 s = 0; s < frame_length; s++) {
      int32 s_in_wave = s + wave_start;
      while (s_in_wave < 0 || s_in_wave >= wave_dim) {
        // reflect around the beginning or end of the wave.
        // e.g. -1 -> 0, -2 -> 1.
        // dim -> dim - 1, dim + 1 -> dim - 2.
        // the code supports repeated reflections, although this
        // would only be needed in pathological cases.
        if (s_in_wave < 0) s_in_wave = - s_in_wave - 1;
        else s_in_wave = 2 * wave_dim - 1 - s_in_wave;
      }
      (*window)(s) = wave(s_in_wave);
    }
  }

  if (frame_length_padded > frame_length)
    window->Range(frame_length, frame_length_padded - frame_length).SetZero();

  SubVector<BaseFloat> frame(*window, 0, frame_length);

  ProcessWindow(opts, window_function, &frame);

}

}  // namespace kaldi

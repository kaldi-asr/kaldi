// feat/feature-functions.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation

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



#include "feat/feature-functions.h"
#include "feat/feature-functions.h"
#include "matrix/matrix-functions.h"

//#define DEBUG

namespace kaldi {


int32 NumFrames(size_t nsamp,
                const FrameExtractionOptions &opts) {
  int32 frame_shift = opts.WindowShift();
  int32 frame_length = opts.WindowSize();
  assert(frame_shift != 0 && frame_length != 0);
  if (static_cast<int32>(nsamp) < frame_length) return 0;
  else return (1 + ((nsamp - frame_length) / frame_shift));
  // view the expression above as: nsamp-frame_length is how much room we have to shift the
  // frame within the waveform; frame_shift is how much we shift it each time and the ratio
  // is how many times we can shift it (integer arithmetic rounds down).
}


void Dither(VectorBase<BaseFloat> *waveform, BaseFloat dither_value) {
  for (int32 i = 0; i<waveform->Dim(); i++)
    (*waveform)(i) += RandGauss()*dither_value;
}


void Preemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  assert(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
  for (int32 i = waveform->Dim()-1; i > 0; i--)
    (*waveform)(i) -= preemph_coeff * (*waveform)(i-1);
  (*waveform)(0) -= preemph_coeff * (*waveform)(0);
}



FeatureWindowFunction::FeatureWindowFunction(const FrameExtractionOptions &opts) {
  size_t frame_length = opts.WindowSize();
  assert(frame_length > 0);
  window.Resize(frame_length);
  for (size_t i = 0; i < frame_length; i++) {
    if (opts.window_type == "hanning") {
      window(i) = 0.5  - 0.5*cos(M_2PI * static_cast<BaseFloat>(i) / (frame_length-1));
    } else if (opts.window_type == "hamming") {
      window(i) = 0.54 - 0.46*cos(M_2PI * static_cast<BaseFloat>(i) / (frame_length-1));
    } else if (opts.window_type == "povey") {  // like hamming but goes to zero at edges.
      window(i) = pow(0.5  - 0.5*cos(M_2PI * static_cast<BaseFloat>(i) / (frame_length-1)), 0.85);
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
  assert(window_function.window.Dim() == frame_length);
  assert(frame_shift != 0 && frame_length != 0);
  int32 start = frame_shift*f, end = start + frame_length;
  assert(start >= 0 && end <= wave.Dim());
  assert(window != NULL);
  int32 frame_length_padded = opts.PaddedWindowSize();

  if (window->Dim() != frame_length_padded)
    window->Resize(frame_length_padded);
  SubVector<BaseFloat> wave_part(wave, start, frame_length);

  SubVector<BaseFloat> window_part(*window, 0, frame_length);
  window_part.CopyFromVec(wave_part);

  if (opts.dither != 0.0) Dither(&window_part, opts.dither);

  if (opts.remove_dc_offset != 0.0)
    window_part.Add(-window_part.Sum() / frame_length);

  if (log_energy_pre_window != NULL) {
    BaseFloat energy = VecVec(wave_part, wave_part);
    *log_energy_pre_window = log(energy);
  }

  if (opts.preemph_coeff != 0.0)
    Preemphasize(&window_part, opts.preemph_coeff);

  window_part.MulElements(window_function.window);  // Multiply by windowing function.

  if (frame_length != frame_length_padded)
    SubVector<BaseFloat>(*window, frame_length, frame_length_padded-frame_length).SetZero();
}

void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder) {
  int32 frame_shift = opts.WindowShift();
  int32 num_frames = NumFrames(wave.Dim(), opts);
  int32 offset = num_frames * frame_shift;  // this is the amount at the start that has
  // been consumed.
  assert(wave_remainder != NULL);
  int32 remaining_len = wave.Dim() - offset;
  wave_remainder->Resize(remaining_len);
  assert(remaining_len >= 0);
  if (remaining_len > 0)
    wave_remainder->CopyFromVec(SubVector<BaseFloat>(wave, offset, remaining_len));
}


void ComputePowerSpectrum(VectorBase<BaseFloat> *waveform) {
  int32 dim = waveform->Dim();

  // no, letting it be non-power-of-two for now.
  // assert(dim > 0 && (dim & (dim-1) == 0));  // make sure a power of two.. actually my FFT code
  // does not require this (dan) but this is better in case we use different code [dan].

  // RealFft(waveform, true);  // true == forward (not inverse) FFT; makes no difference here,
  // as we just want power spectrum.

  // now we have in waveform, first half of complex spectrum
  // it's stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32 half_dim = dim/2;
  BaseFloat first_energy = (*waveform)(0) * (*waveform)(0),
      last_energy = (*waveform)(1) * (*waveform)(1);  // handle this special case
  for (int32 i = 1; i < half_dim; i++) {
    BaseFloat real = (*waveform)(i*2), im = (*waveform)(i*2 + 1);
    (*waveform)(i) = real*real + im*im;
  }
  (*waveform)(0) = first_energy;
  (*waveform)(half_dim) = last_energy;  // Will actually never be used, and anyway
  // if the signal has been bandlimited sensibly this should be zero.
}


DeltaFeatures::DeltaFeatures(const DeltaFeaturesOptions &opts): opts_(opts) {
  assert(opts.order >= 0 && opts.order < 1000);  // just make sure we don't get binary junk.
  // opts will normally be 2 or 3.
  assert(opts.window > 0 && opts.window < 1000);  // again, basic sanity check.
  // normally the window size will be two.

  scales_.resize(opts.order+1);
  scales_[0].Resize(1);
  scales_[0](0) = 1.0;  // trivial window for 0th order delta [i.e. baseline feats]

  for (int32 i = 1; i <= opts.order; i++) {
    Vector<BaseFloat> &prev_scales = scales_[i-1],
        &cur_scales = scales_[i];
    int32 window = opts.window;  // this code is designed to still
    // work if instead we later make it an array and do opts.window[i-1],
    // or something like that. "window" is a parameter specifying delta-window
    // width which is actually 2*window + 1.
    assert(window != 0);
    int32 prev_offset = (static_cast<int32>(prev_scales.Dim()-1))/2,
        cur_offset = prev_offset + window;
    cur_scales.Resize(prev_scales.Dim() + 2*window);  // also zeros it.

    BaseFloat normalizer = 0.0;
    for (int32 j = -window; j <= window; j++) {
      normalizer += j*j;
      for (int32 k = -prev_offset; k <= prev_offset; k++) {
        cur_scales(j+k+cur_offset) +=
            static_cast<BaseFloat>(j) * prev_scales(k+prev_offset);
      }
    }
    cur_scales.Scale(1.0 / normalizer);
  }
}

void DeltaFeatures::Process(const MatrixBase<BaseFloat> &input_feats,
                            int32 frame,
                            SubVector<BaseFloat> *output_frame) const {
  assert(frame < input_feats.NumRows());
  int32 num_frames = input_feats.NumRows(),
      feat_dim = input_feats.NumCols();
  assert(static_cast<int32>(output_frame->Dim()) == feat_dim * (opts_.order+1));
  output_frame->SetZero();
  for (int32 i = 0; i <= opts_.order; i++) {
    const Vector<BaseFloat> &scales = scales_[i];
    int32 offset = (scales.Dim()-1)/2;
    SubVector<BaseFloat> output(*output_frame, i*feat_dim, feat_dim);
    for (int32 j = -offset; j <= offset; j++) {
      // if asked to read
      int32 offset_frame = frame + j;
      if (offset_frame < 0) offset_frame = 0;
      else if (offset_frame >= num_frames)
        offset_frame = num_frames-1;
      BaseFloat scale = scales(j+offset);
      if (scale != 0.0)
        output.AddVec(scale, input_feats.Row(offset_frame));
    }
  }
}

void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features) {
  output_features->Resize(input_features.NumRows(),
                          input_features.NumCols()
                          *(delta_opts.order + 1));
  DeltaFeatures delta(delta_opts);
  for (int32 r = 0; r < static_cast<int32>(input_features.NumRows()); r++) {
    SubVector<BaseFloat> row(*output_features, r);
    delta.Process(input_features, r, &row);
  }
}







void GetEqualLoudnessVector(const MelBanks &mel_banks,
                            Vector<BaseFloat> *ans) {
  size_t n = mel_banks.NumBins();
  // central freq of each mel bin
  const Vector<BaseFloat> &f0 = mel_banks.GetCenterFreqs();
  ans->Resize(n);
  for (size_t i = 0; i < n; i++) {
    BaseFloat fsq = f0(i) * f0(i);
    BaseFloat fsub = fsq / (fsq + 1.6e5);
    (*ans)(i) = fsub * fsub * ((fsq + 1.44e6) / (fsq + 9.61e6));
  }
}

void InitIdftBases(size_t n_bases, size_t dimension, Matrix<BaseFloat>* mat_out) {

  float angle = M_PI / (float)(dimension - 1);
  float scale = 1.0f / (2.0f * (dimension - 1));
  size_t i, j;
  mat_out->Resize(n_bases, dimension);
  for (i = 0; i < n_bases; i++)
  {
    (*mat_out)(i, 0) = 1.0f * scale;
    for (j = 1; j < dimension - 1; j++)
       (*mat_out)(i, j) = 2.0 * scale * cos(angle * (float)i * (float)j);

    (*mat_out)(i, dimension -1) = scale * cos(angle * (float)i * (float)(dimension-1));
  }
}

// Compute LP coefficients from autocorrelation coefficients.
BaseFloat ComputeLpc(const VectorBase<BaseFloat> &autocorr_in,
                     Vector<BaseFloat> *lpc_out) {
  int32 n = autocorr_in.Dim() - 1;
  assert(lpc_out->Dim() == n);
  Vector<BaseFloat> tmp(n);
  BaseFloat ans =  Durbin(n, autocorr_in.Data(),
                          lpc_out->Data(),
                          tmp.Data());
  if (ans <= 0.0)
    KALDI_WARN << "Zero energy in LPC computation";
  return -log((double)1.0/ans);  // forms the C0 value
}

void SpliceFrames(const MatrixBase<BaseFloat> &input_features,
                  int32 left_context,
                  int32 right_context,
                  Matrix<BaseFloat> *output_features) {
  int32 T = input_features.NumRows(), D = input_features.NumCols();
  if (T == 0 || D == 0)
    KALDI_ERR << "SpliceFrames: empty input\n";
  KALDI_ASSERT(left_context >= 0 && right_context >= 0);
  int32 N = 1 + left_context + right_context;
  output_features->Resize(T, D*N);
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> dst_row(*output_features, t);
    for (int32 j = 0; j < N; j++) {
      int32 t2 = t + j - left_context;
      if (t2 < 0) t2 = 0;
      if (t2 >= T) t2 = T-1;
      SubVector<BaseFloat> dst(dst_row, j*D, D),
          src(input_features, t2);
      dst.CopyFromVec(src);
    }
  }
}


} // namespace kaldi

// feat/pitch-functions.cc

// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer

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

#include <algorithm>
#include <limits>
#include "feat/feature-functions.h"
#include "matrix/matrix-functions.h"
#include "feat/pitch-functions.h"
#include "feat/resample.h"
#include "feat/mel-computations.h"

namespace kaldi {


/*
  WeightedMovingWindowNormalize does weighted moving window normalization.

  The simplest possible moving window normalization would be to set
  mean_subtracted_log_pitch(i) to raw_log_pitch(i) minus
  the average over the range of raw_log_pitch over the range
  [ i - window-size/2 ... i + window-size/2 ].  At the edges of
  the file, the window is truncated to be within the file.

  Weighted moving window normalization subtracts a weighted
  average, where the weight corresponds to "pov" (the probability
  of voicing).  This seemed to slightly improve results versus
  vanilla moving window normalization, although the effect was small.
*/

void WeightedMovingWindowNormalize(
    int32 normalization_window_size,
    const VectorBase<BaseFloat> &pov,
    const VectorBase<BaseFloat> &raw_log_pitch,
    Vector<BaseFloat> *normalized_log_pitch) {
  int32 num_frames = pov.Dim();
  KALDI_ASSERT(num_frames == raw_log_pitch.Dim());
  int32 last_window_start = -1, last_window_end = -1;
  double weighted_sum = 0.0, pov_sum = 0.0;

  for (int32 t = 0; t < num_frames; t++) {
    int32 window_start, window_end;
    window_start = t - (normalization_window_size/2);
    window_end = window_start + normalization_window_size;

    if (window_start < 0) {
      window_end -= window_start;
      window_start = 0;
    }

    if (window_end > num_frames) {
      window_start -= (window_end - num_frames);
      window_end = num_frames;
      if (window_start < 0) window_start = 0;
    }
    if (last_window_start == -1) {
      SubVector<BaseFloat> pitch_part(raw_log_pitch, window_start,
                                      window_end - window_start);
      SubVector<BaseFloat> pov_part(pov, window_start,
                                    window_end - window_start);
      // weighted sum of pitch
      weighted_sum += VecVec(pitch_part, pov_part);

      // sum of pov
      pov_sum = pov_part.Sum();
    } else {
      if (window_start > last_window_start) {
        KALDI_ASSERT(window_start == last_window_start + 1);
        pov_sum -= pov(last_window_start);
        weighted_sum -= pov(last_window_start)*raw_log_pitch(last_window_start);
      }
      if (window_end > last_window_end) {
        KALDI_ASSERT(window_end == last_window_end + 1);
        pov_sum += pov(last_window_end);
        weighted_sum += pov(last_window_end) * raw_log_pitch(last_window_end);
      }
    }

    KALDI_ASSERT(window_end - window_start > 0);
    last_window_start = window_start;
    last_window_end = window_end;
    (*normalized_log_pitch)(t) = raw_log_pitch(t) -
                                      weighted_sum / pov_sum;
    KALDI_ASSERT((*normalized_log_pitch)(t) -
                 (*normalized_log_pitch)(t) == 0);
  }
}

/**
   This function processes the NCCF n to a POV feature f by applying the formula
     f = (1.0001 - n)^0.15  - 1.0.
   This is a nonlinear function designed to make the output reasonably Gaussian
   distributed.  Before doing this, the NCCF distribution is in the range [-1,
   1] but has a strong peak just before 1.0, which this function smooths out.
 */
BaseFloat NccfToPovFeature(BaseFloat n) {
  if (n > 1.0) {
    n = 1.0;
  } else if (n < -1.0) {
    n = -1.0;
  }
  BaseFloat f = pow((1.0001 - n), 0.15) - 1.0;
  KALDI_ASSERT(f - f == 0);  // check for NaN,inf.
  return f;
}

/**
   This function processes the NCCF n to a reasonably accurate probability
   of voicing p by applying the formula:

      n' = fabs(n)
      r = -5.2 + 5.4 * exp(7.5 * (n' - 1.0)) +
           4.8 * n' - 2.0 * exp(-10.0 * n') + 4.2 * exp(20.0 * (n' - 1.0));
      p = 1.0 / (1 + exp(-1.0 * r));

   How did we get this formula?  We plotted the empirical log-prob-ratio of voicing
    r = log( p[voiced] / p[not-voiced] )
   [on the Keele database where voicing is marked], as a function of the NCCF at
   the delay picked by our algorithm.  This was done on intervals of the NCCF, so
   we had enough statistics to get that ratio.  The NCCF covers [-1, 1]; almost
   all of the probability mass is on [0, 1] but the empirical POV seems fairly
   symmetric with a minimum near zero, so we chose to make it a function of n' = fabs(n).
   
   Then we manually tuned a function (the one you see above) that approximated
   the log-prob-ratio of voicing fairly well as a function of the absolute-value
   NCCF n'; however, wasn't a very exact match since we were also trying to make
   the transformed NCCF fairly Gaussian distributed, with a view to using it as
   a feature-- an idea we later abandoned after a simpler formula worked better.
 */
BaseFloat NccfToPov(BaseFloat n) {
  BaseFloat ndash = fabs(n);
  if (ndash > 1.0) ndash = 1.0;  // just in case it was slightly outside [-1, 1]

  BaseFloat r = -5.2 + 5.4 * exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
                2.0 * exp(-10.0 * ndash) + 4.2 * exp(20.0 * (ndash - 1.0));
  // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
  BaseFloat p = 1.0 / (1 + exp(-1.0 * r));
  KALDI_ASSERT(p - p == 0);  // Check for NaN/inf
  return p;
}


int32 PitchNumFrames(int32 nsamp,
                     const PitchExtractionOptions &opts) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  double outer_max_lag =  1.0 / opts.min_f0 +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  int32 end_lag = floor(opts.resample_freq * outer_max_lag);
  int32 frame_length_full = frame_length + end_lag;
  
  KALDI_ASSERT(frame_shift != 0 && frame_length_full != 0);
  if (static_cast<int32>(nsamp) < frame_length_full)
    return 0;
  else
    return (1 + ((nsamp - frame_length_full) / frame_shift));
}

void PreemphasizeFrame(VectorBase<BaseFloat> *waveform, double preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  KALDI_ASSERT(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
  for (int32 i = waveform->Dim()-1; i > 0; i--)
    (*waveform)(i) -= preemph_coeff * (*waveform)(i-1);
  (*waveform)(0) -= preemph_coeff * (*waveform)(0);
}

void ExtractFrame(const VectorBase<BaseFloat> &wave,
                  int32 frame_index,
                  const PitchExtractionOptions &opts,
                  Vector<BaseFloat> *window) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  double outer_max_lag =  1.0 / opts.min_f0 +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  int32 end_lag = floor(opts.resample_freq * outer_max_lag);

  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  int32 start = frame_shift * frame_index;
  int32 frame_length_full = frame_length + end_lag;

  if (window->Dim() != frame_length_full)
    window->Resize(frame_length_full);

  if (start + frame_length_full <= wave.Dim()) {
    window->CopyFromVec(wave.Range(start, frame_length_full));
  } else {
    window->SetZero();
    int32 size = wave.Dim() - start;
    window->Range(0, size).CopyFromVec(wave.Range(start, size));
  }

  if (opts.preemph_coeff != 0.0)
    PreemphasizeFrame(window, opts.preemph_coeff);
}


void PreProcess(const PitchExtractionOptions opts,
                const VectorBase<BaseFloat> &wave,
                Vector<BaseFloat> *processed_wave) {
  KALDI_ASSERT(processed_wave != NULL);
  // down-sample and Low-Pass filtering the input wave
  int32 num_samples_in = wave.Dim();
  double dt = opts.samp_freq / opts.resample_freq;
  int32 resampled_len = 1 + static_cast<int>(num_samples_in / dt);
  processed_wave->Resize(resampled_len);  // filtered wave
  LinearResample resample(opts.samp_freq, opts.resample_freq,
                          opts.lowpass_cutoff,
                          opts.lowpass_filter_width);
  const bool flush = true;
  resample.Resample(wave, flush, processed_wave);

  // Normalize input signal using rms
  double rms = pow(VecVec((*processed_wave), (*processed_wave))
                   / processed_wave->Dim(), 0.5);
  if (rms != 0.0)
    (*processed_wave).Scale(1.0 / rms);
}


/**
   This function computes some dot products that are required
   while computing the NCCF.
   For each integer lag from start to end-1, this function
   outputs to (*inner_prod)(lag - start), the dot-product
   of a window starting at 0 with a window starting at
   lag.  All windows are of length nccf_window_size.  It
   outputs to (*norm_prod)(lag - start), e1 * e2, where
   e1 is the dot-product of the un-shifted window with itself,
   and d2 is the dot-product of the window shifted by "lag"
   with itself.
 */
void ComputeCorrelation(const VectorBase<BaseFloat> &wave,
                        int32 first_lag, int32 last_lag,
                        int32 nccf_window_size,
                        VectorBase<BaseFloat> *inner_prod,
                        VectorBase<BaseFloat> *norm_prod) {
  Vector<BaseFloat> zero_mean_wave(wave);
  // TODO: possibly fix this, the mean normalization is done in a strange way.
  SubVector<BaseFloat> wave_part(wave, 0, nccf_window_size);
  // subtract mean-frame from wave
  zero_mean_wave.Add(-wave_part.Sum() / nccf_window_size);
  BaseFloat e1, e2, sum;
  SubVector<BaseFloat> sub_vec1(zero_mean_wave, 0, nccf_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  for (int32 lag = first_lag; lag <= last_lag; lag++) {
    SubVector<BaseFloat> sub_vec2(zero_mean_wave, lag, nccf_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*inner_prod)(lag - first_lag) = sum;
    (*norm_prod)(lag - first_lag) = e1 * e2;
  }
}

/**
   Computes the NCCF as a fraction of the numerator term (a dot product between
   two vectors) and a denominator term which equals sqrt(e1*e2 + nccf_ballast)
   where e1 and e2 are both dot-products of bits of the wave with themselves,
   and e1*e2 is supplied as "norm_prod".  These quantities are computed by
   "ComputeCorrelation".
*/
void ComputeNccf(const VectorBase<BaseFloat> &inner_prod,
                 const VectorBase<BaseFloat> &norm_prod,
                 double nccf_ballast,
                 VectorBase<BaseFloat> *nccf_vec) {
  KALDI_ASSERT(inner_prod.Dim() == norm_prod.Dim() &&
               inner_prod.Dim() == nccf_vec->Dim());
  for (int32 lag = 0; lag < inner_prod.Dim(); lag++) {
    BaseFloat numerator = inner_prod(lag),
        denominator = pow(norm_prod(lag) + nccf_ballast, 0.5),
        nccf;
    if (denominator != 0.0) {
      nccf = numerator / denominator;
    } else {
      KALDI_ASSERT(numerator == 0.0);
      nccf = 0.0;
    }
    KALDI_ASSERT(nccf < 1.01 && nccf > -1.01);
    (*nccf_vec)(lag) = nccf;
  }
}

/**
   This function selects the lags at which we measure the NCCF: we need
   to select lags from 1/max_f0 to 1/min_f0, in a geometric progression
   with ratio 1 + d.
 */
void SelectLags(const PitchExtractionOptions &opts,
                Vector<BaseFloat> *lags) {
  // choose lags relative to acceptable pitch tolerance
  BaseFloat min_lag = 1.0 / opts.max_f0, max_lag = 1.0 / opts.min_f0;

  std::vector<BaseFloat> tmp_lags;
  for (BaseFloat lag = min_lag; lag <= max_lag; lag *= 1.0 + opts.delta_pitch)
    tmp_lags.push_back(lag);
  lags->Resize(tmp_lags.size());
  std::copy(tmp_lags.begin(), tmp_lags.end(), lags->Data());
}

class PitchExtractor {
 public:
  PitchExtractor(const PitchExtractionOptions &opts,
                 const Vector<BaseFloat> &lags,
                 int32 num_states,
                 int32 num_frames) :
      opts_(opts),
      num_states_(num_states),
      num_frames_(num_frames),
      lags_(lags) {
    frames_.resize(num_frames_+1);
    for (int32 i = 0; i < num_frames_+1; i++) {
      frames_[i].obj_func.Resize(num_states_);
      frames_[i].backpointers.resize(num_states_);
    }
  }
  ~PitchExtractor() {}

  void ComputeLocalCost(const VectorBase<BaseFloat> &nccf_row,
                        VectorBase<BaseFloat> *local_cost) {
    // compute the local cost
    local_cost->Set(1.0);
    local_cost->AddVec(-1.0, nccf_row);
    Vector<BaseFloat> corr_lag_cost(num_states_);
    corr_lag_cost.AddVecVec(opts_.soft_min_f0, nccf_row, lags_, 0);
    local_cost->AddVec(1.0, corr_lag_cost);
  }

  void FastViterbi(const Matrix<BaseFloat> &nccf) {
    double intercost, min_c, this_c;
    int best_b, min_i, max_i;
    BaseFloat delta_pitch_sq = log(1 + opts_.delta_pitch)
      * log(1 + opts_.delta_pitch);
    // loop over frames
    for (int32 t = 1; t < num_frames_ + 1; t++) {
      // The stuff with the "forward pass" and "backward "pass" is described in
      // the paper; it's an algorithm that allows us to compute the vector of
      // forward-costs and back-pointers without accessing all of the pairs of
      // [pitch on last frame, pitch on this frame].
      Vector<BaseFloat> local_cost(num_states_);
      ComputeLocalCost(nccf.Row(t - 1), &local_cost);
      // Forward Pass
      for (int32 i = 0; i < num_states_; i++) {
        if ( i == 0 )
          min_i = 0;
        else
          min_i = frames_[t].backpointers[i-1];
        min_c = std::numeric_limits<double>::infinity();
        best_b = -1;

        for (int32 k = min_i; k <= i; k++) {
          intercost = (i-k) * (i-k) * delta_pitch_sq;
          this_c = frames_[t-1].obj_func(k)+ opts_.penalty_factor * intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].backpointers[i] = best_b;
        frames_[t].obj_func(i) = min_c + local_cost(i);
      }
      // Backward Pass
      for (int32 i = num_states_-1; i >= 0; i--) {
        if (i == num_states_-1)
          max_i = num_states_-1;
        else
          max_i = frames_[t].backpointers[i+1];
        min_c = frames_[t].obj_func(i) - local_cost(i);
        best_b = frames_[t].backpointers[i];

        for (int32 k = i+1 ; k <= max_i; k++) {
          intercost = (i-k) * (i-k) * delta_pitch_sq;
          this_c = frames_[t-1].obj_func(k)+ opts_.penalty_factor *intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].backpointers[i] = best_b;
        frames_[t].obj_func(i) = min_c + local_cost(i);
      }
    }
    {
      int32 num_frames = frames_.size() - 1;
      KALDI_VLOG(3) << "Viterbi cost is " 
                    << (frames_.back().obj_func.Min() / num_frames)
                    << " per frame, over " << num_frames << " frames.";
    }
  }

  void FindBestPath(const Matrix<BaseFloat> &correlation) {
    // Find the Best path using backpointers
    int32 i = num_frames_;
    int32 best;
    double l_opt;
    frames_[i].obj_func.Min(&best);
    while (i > 0) {
      l_opt = lags_(best);
      frames_[i].truepitch = 1.0 / l_opt;
      frames_[i].pov = correlation(i-1, best);
      best = frames_[i].backpointers[best];
      i--;
    }
  }
  void GetPitchAndPov(Matrix<BaseFloat> *output) {
    output->Resize(num_frames_, 2);
    for (int32 frm = 0; frm < num_frames_; frm++) {
      (*output)(frm, 0) = static_cast<BaseFloat>(frames_[frm + 1].pov);
      (*output)(frm, 1) = static_cast<BaseFloat>(frames_[frm + 1].truepitch);
    }
  }
 private:
  PitchExtractionOptions opts_;
  int32 num_states_;    // number of states in Viterbi Computation
  int32 num_frames_;    // number of frames in input wave
  Vector<BaseFloat> lags_;    // all lags used in viterbi
  struct PitchFrame {
    Vector<double> obj_func;      // optimal objective function for frame i
    std::vector<int32> backpointers;
    double truepitch;             // True pitch
    double pov;                   // probability of voicing
    explicit PitchFrame() {}
  };
  std::vector< PitchFrame > frames_;
};

void ComputeKaldiPitch(const PitchExtractionOptions &opts,
                       const VectorBase<BaseFloat> &wave,
                       Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);

  // Preprocess the wave
  Vector<BaseFloat> processed_wave(wave.Dim());
  PreProcess(opts, wave, &processed_wave);
  int32 num_frames = PitchNumFrames(processed_wave.Dim(), opts);
  if (num_frames == 0)
    KALDI_ERR << "No frames fit in file (#samples is "
      << processed_wave.Dim() << ")";
  Vector<BaseFloat> window;       // windowed waveform.
  double outer_min_lag = 1.0 / opts.max_f0 -
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / opts.min_f0 +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));

  int32 start = ceil(opts.resample_freq * outer_min_lag),
      end = floor(opts.resample_freq * outer_max_lag),
      num_initial_lags = end - start + 1;
  // num_initial_lags is the number of lags we initially compute-- evenly
  // spaced, integer lags-- before doing the resampling.

  Vector<BaseFloat> final_lags;
  SelectLags(opts, &final_lags);
  // final_lags is a list of the lags we want to resample the NCCF at;
  // these are log-spaced so we have finer resolution for smaller lags.

  int32 num_states = final_lags.Dim();
  Matrix<BaseFloat> nccf_pitch(num_frames, num_initial_lags),
      nccf_pov(num_frames, num_initial_lags);

  double nccf_ballast_pitch = pow(opts.NccfWindowSize(), 2) * opts.nccf_ballast,
      nccf_ballast_pov = 0.0;

  for (int32 r = 0; r < num_frames; r++) {  // r is frame index.
    ExtractFrame(processed_wave, r, opts, &window);
    // compute nccf for pitch extraction
    Vector<BaseFloat> inner_prod(num_initial_lags), norm_prod(num_initial_lags);
    ComputeCorrelation(window, start, end, opts.NccfWindowSize(),
                       &inner_prod, &norm_prod);
    SubVector<BaseFloat> nccf_pitch_vec(nccf_pitch.Row(r));
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pitch,
                &(nccf_pitch_vec));
    // compute the Nccf for Probability of voicing estimation;
    // there is no ballast for this version of the NCCF.
    SubVector<BaseFloat> nccf_pov_vec(nccf_pov.Row(r));
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pov,
                &(nccf_pov_vec));
  }
  Vector<BaseFloat> final_lags_offset(final_lags);
  // final_lags_offset is the final_lags with start / opts.resample_freq
  // subtracted from each element, so we can treat the rows of nccf_pitch and
  // nccf_pov as starting from sample zero in a signal that starts at the point
  // start / opts.resample_freq.  This is necessary because the resampling code
  // assumes that the input signal starts from sample zero.
  final_lags_offset.Add(-start / opts.resample_freq);

  // upsample_cutoff is the filter cutoff for upsampling the NCCF, which is the
  // Nyquist of the resampling frequency.  The NCCF is (almost completely)
  // bandlimited to around "lowpass_cutoff" (1000 by default), and when the
  // spectrum of this bandlimited signal is convolved with the spectrum of an
  // impulse train with frequency "resample_freq", which are separated by 4kHz,
  // we get energy at -5000,-3000, -1000...1000, 3000..5000, etc.  Filtering at
  // half the Nyquist (2000 by default) is sufficient to get only the first
  // repetition.
  BaseFloat upsample_cutoff = opts.resample_freq * 0.5;
  ArbitraryResample resample(num_initial_lags, opts.resample_freq,
                             upsample_cutoff, final_lags_offset,
                             opts.upsample_filter_width);
  Matrix<BaseFloat> resampled_nccf_pitch(num_frames, num_states);
  resample.Resample(nccf_pitch, &resampled_nccf_pitch);
  Matrix<BaseFloat> resampled_nccf_pov(num_frames, num_states);
  resample.Resample(nccf_pov, &resampled_nccf_pov);


  if (resampled_nccf_pitch.NumRows() > 100) {
    KALDI_VLOG(4) << "frame 100: nccf_pitch is "
                  << nccf_pitch.Row(100);
    
    KALDI_VLOG(4) << "frame 100: nccf_pitch_resampled is "
                  << resampled_nccf_pitch.Row(100);
  }
  

  PitchExtractor pitch(opts, final_lags, num_states, num_frames);
  pitch.FastViterbi(resampled_nccf_pitch);
  // pitch.FindBestPath will use the NCCF without the "ballast" term
  // when it notes the NCCF at each frame.
  pitch.FindBestPath(resampled_nccf_pov);
  output->Resize(num_frames, 2);  // (pov, pitch)
  pitch.GetPitchAndPov(output);
}

/**
   This function applies to the pitch the normal delta (time-derivative)
   computation using a five frame window, multiplying by a normalized version of
   the scales [ -2, 1, 0, 1, 2 ].  It then adds a small amount of noise to the
   output, in order to avoid peaks appearing in the distribution of delta pitch,
   that correspond to the discretization interval for log-pitch.
*/
void ExtractDeltaPitch(const PostProcessPitchOptions &opts,
                       const VectorBase<BaseFloat> &input,
                       Vector<BaseFloat> *output) {
  int32 num_frames = input.Dim();
  DeltaFeaturesOptions delta_opts;
  delta_opts.order = 1;
  delta_opts.window = opts.delta_window;
  Matrix<BaseFloat> matrix_input(num_frames, 1),
      matrix_output;
  matrix_input.CopyColFromVec(input, 0);
  ComputeDeltas(delta_opts, matrix_input, &matrix_output);
  KALDI_ASSERT(matrix_output.NumRows() == matrix_input.NumRows() &&
               matrix_output.NumCols() == 2);
  output->Resize(num_frames);
  output->CopyColFromMat(matrix_output, 1);

  Vector<BaseFloat> noise(num_frames);
  noise.SetRandn();
  output->AddVec(opts.delta_pitch_noise_stddev, noise);
}


void PostProcessPitch(const PostProcessPitchOptions &opts,
                      const MatrixBase<BaseFloat> &input,
                      Matrix<BaseFloat> *output) {
  int32 T = input.NumRows();
  // We've coded this for clarity rather than memory efficiency; anyway the
  // memory consumption is trivial.
  Vector<BaseFloat> nccf(T), raw_pitch(T), raw_log_pitch(T),
      pov(T), pov_feature(T), normalized_log_pitch(T),
      delta_log_pitch(T);

  nccf.CopyColFromMat(input, 0);
  raw_pitch.CopyColFromMat(input, 1);
  KALDI_ASSERT(raw_pitch.Min() > 0 && "Non-positive pitch.");
  raw_log_pitch.CopyFromVec(raw_pitch);
  raw_log_pitch.ApplyLog();
  for (int32 t = 0; t < T; t++) {
    pov(t) = NccfToPov(nccf(t));
    pov_feature(t) = opts.pov_scale * NccfToPovFeature(nccf(t));
  }
  WeightedMovingWindowNormalize(opts.normalization_window_size,
                                pov, raw_log_pitch, &normalized_log_pitch);
  // the normalized log pitch has quite a small variance; scale it up a little
  // (this interacts with variance flooring in early system build stages).
  normalized_log_pitch.Scale(opts.pitch_scale);

  ExtractDeltaPitch(opts, raw_log_pitch, &delta_log_pitch);
  delta_log_pitch.Scale(opts.delta_pitch_scale);

  // Normally we'll have all of these but raw_log_pitch.
  int32 output_ncols =
      (opts.add_pov_feature ? 1 : 0) +
      (opts.add_normalized_log_pitch ? 1 : 0) +
      (opts.add_delta_pitch ? 1 : 0) +
      (opts.add_raw_log_pitch ? 1 : 0);
  if (output_ncols == 0) {
    KALDI_ERR << " At least one of the pitch features should be chosen. "
              << "Check your post-process pitch options.";
  }
  output->Resize(T, output_ncols, kUndefined);
  int32 col = 0;
  if (opts.add_pov_feature)
    output->CopyColFromVec(pov_feature, col++);
  if (opts.add_normalized_log_pitch)
    output->CopyColFromVec(normalized_log_pitch, col++);
  if (opts.add_delta_pitch)
    output->CopyColFromVec(delta_log_pitch, col++);
  if (opts.add_raw_log_pitch)
    output->CopyColFromVec(raw_log_pitch, col++);
  KALDI_ASSERT(col == output_ncols);
}

/**
   This function computes the local-cost for the Viterbi computation,
   see eq. (5) in the paper.
   @param  opts         The options as provided by the user
   @param  nccf_pitch   The nccf as computed for the pitch computation (with ballast).
   @param  lags         The log-spaced lags at which nccf_pitch is sampled.
   @param  local_cost   We output the local-cost to here.
*/
void ComputeLocalCost(const VectorBase<BaseFloat> &nccf_pitch,
                      const VectorBase<BaseFloat> &lags,
                      const PitchExtractionOptions &opts,
                      VectorBase<BaseFloat> *local_cost) {
  // from the paper, eq. 5, local_cost = 1 - Phi(t,i)(1 - soft_min_f0 L_i)
  // nccf is the nccf on this frame measured at the lags in "lags".
  KALDI_ASSERT(nccf_pitch.Dim() == local_cost->Dim() &&
               nccf_pitch.Dim() == lags.Dim());
  local_cost->Set(1.0);
  // add the term -Phi(t,i):
  local_cost->AddVec(-1.0, nccf_pitch);
  // add the term soft_min_f0 Phi(t,i) L_i
  local_cost->AddVecVec(opts.soft_min_f0, lags, nccf_pitch, 1.0); 
}



// class PitchFrameInfo is used inside class OnlinePitchFeatureImpl.
// It stores the information we need to keep around for a single frame
// of the pitch computation.
class PitchFrameInfo {
 public:
  /// This function resizes the arrays for this object and updates the reference
  /// counts for the previous object (by decrementing those reference counts
  /// when we destroy a StateInfo object).  A StateInfo object is considered to
  /// be destroyed when we delete it, not when its reference counts goes to
  /// zero.
  void Cleanup(PitchFrameInfo *prev_frame);

  /// This function may be called for the last (most recent) PitchFrameInfo object
  /// with the best state (obtained from the externally held forward-costs).  It
  /// traces back as far as needed to set the cur_best_state_, and as it's
  /// going it sets the lag-index and pov_nccf in pitch_pov_iter, which
  /// when it's called is an iterator to where to put the info for the final state;
  /// the iterator will be decremented inside this function.
  void SetBestState(int32 best_state,
                    std::vector<std::pair<int32, BaseFloat> >::iterator lag_nccf_iter);
                    
                    

  /// This function may be called on the last (most recent) PitchFrameInfo
  /// object; it computes how many frames of latency there is because the
  /// traceback has not yet settled on a single value for frames in the past.
  /// It actually returns the minimum of max_latency and the actual latency,
  /// which is an optimization because we won't care about latency past
  /// a user-specified maximum latency.
  int32 ComputeLatency(int32 max_latency);
  
  /// This function updates
  bool UpdatePreviousBestState(PitchFrameInfo *prev_frame);

  /// This constructor is used for frame -1; it sets the costs to be all zeros
  /// the pov_nccf's to zero and the backpointers to -1.
  PitchFrameInfo(int32 num_states);


  /// This constructor is used for frames apart from frame -1; the bulk of
  /// the Viterbi computation takes place inside this constructor.
  ///  @param  opts         The options as provided by the user
  ///  @param  nccf_pitch   The nccf as computed for the pitch computation (with ballast).
  ///  @param  nccf_pov     The nccf as computed for the POV computation (without ballast).
  ///  @param  lags         The log-spaced lags at which nccf_pitch and nccf_pov are sampled.
  ///  @param  prev_frame_forward_cost   The forward-cost vector for the previous frame.
  ///  @param  this_forward_cost   The forward-cost vector for this frame (to be computed).
  PitchFrameInfo(const PitchExtractionOptions &opts,
                 const VectorBase<BaseFloat> &nccf_pitch, 
                 const VectorBase<BaseFloat> &nccf_pov,
                 const VectorBase<BaseFloat> &lags,
                 const VectorBase<double> &prev_forward_cost,
                 PitchFrameInfo *prev_info,
                 VectorBase<double> *this_forward_cost);
 private:
  // struct StateInfo is the information we keep for a single one of the log-spaced
  // lags, for a single frame.  This is a state in the Viterbi computation.
  struct StateInfo {
    /// The state index on the previous frame that is the best preceding state
    /// for this state.
    int32 backpointer;
    /// the version of the NCCF we keep for the POV computation (without the
    /// ballast term).
    BaseFloat pov_nccf;
    StateInfo(): backpointer(0), pov_nccf(0.0) { }
  };
  std::vector<StateInfo> state_info_;
  /// the state index of the first entry in "state_info"; this will initially be
  /// zero, but after cleanup might be nonzero.
  int32 state_offset_;

  /// The current best state in the backtrace from the end.
  int32 cur_best_state_;

  /// The structure for the previous frame.
  PitchFrameInfo *prev_info_;
};


// This constructor is used for frame -1; it sets the costs to be all zeros
// the pov_nccf's to zero and the backpointers to -1.
PitchFrameInfo::PitchFrameInfo(int32 num_states):
    state_info_(num_states), state_offset_(0),
    cur_best_state_(-1), prev_info_(NULL) { } 


PitchFrameInfo::PitchFrameInfo(const PitchExtractionOptions &opts,
                               const VectorBase<BaseFloat> &nccf_pitch, 
                               const VectorBase<BaseFloat> &nccf_pov,
                               const VectorBase<BaseFloat> &lags,
                               const VectorBase<double> &prev_forward_cost,
                               PitchFrameInfo *prev_info,
                               VectorBase<double> *this_forward_cost):
    state_info_(nccf_pitch.Dim()), state_offset_(0), cur_best_state_(-1),
    prev_info_(prev_info) {
  int32 num_states = nccf_pitch.Dim();
  
  Vector<BaseFloat> local_cost(num_states, kUndefined);
  ComputeLocalCost(nccf_pitch, lags, opts, &local_cost);

  const double delta_pitch_sq = pow(log(1.0 + opts.delta_pitch), 2.0),
      inter_frame_factor = delta_pitch_sq * opts.penalty_factor;

  // The algorithm has a forward pass and a backward pass, as briefly described
  // in  the paper.
  // We modified it for additional efficiency, to first compute every Nth frame
  // using the forward and backward passes and then fill in the ones in between.
  // the general principle is that the backpointer for state i must be >= the
  // backpointer for state j, if j >= i.  There is no inexactness.
  
  const int32 modulus = 4;
  int32 i;
  // Forward Pass over every Nth frame  
  for (i = 0; i < num_states; i += modulus) {
    int32 min_i = (i == 0 ? 0 : state_info_[i - modulus].backpointer);
    double min_cost = std::numeric_limits<double>::infinity();
    double best_backpointer = -1;
    
    for (int32 k = min_i; k <= i; k++) {
      double inter_frame_cost = (k - i) * (k - i) * inter_frame_factor;
      double this_cost = prev_forward_cost(k) + inter_frame_cost;
      if (this_cost < min_cost) {
        min_cost = this_cost;
        best_backpointer = k;
      }
    }
    // note: the backpointer and forward-cost are not final, they might change
    // in the backward pass.
    state_info_[i].backpointer = best_backpointer;
    // the forward cost does not get the cocal cost included until now.
    (*this_forward_cost)(i) = min_cost + local_cost(i);
  }

  // Backward Pass over every Nth frame
  int32 last_i = i - modulus; // the last one we processed in previous loop.
  for (i = last_i; i >= 0; i -= modulus) {
    int32 max_i = (i == last_i ?
                   num_states - 1 : state_info_[i + modulus].backpointer);
    double min_cost = (*this_forward_cost)(i) - local_cost(i);
    int32 best_backpointer = state_info_[i].backpointer;
    
    for (int32 k = i + 1 ; k <= max_i; k++) {
      double inter_frame_cost = (k - i) * (k - i) * inter_frame_factor;
      double this_cost = prev_forward_cost(k) + inter_frame_cost;
      if (this_cost < min_cost) {
        min_cost = this_cost;
        best_backpointer = k;
      }
    }
    state_info_[i].backpointer = best_backpointer;
    (*this_forward_cost)(i) = min_cost + local_cost(i);
  }

  // Fill in the frames in between every Nth frame.
  for (int32 i = 0; i < num_states; i++) {
    if (i % modulus != 0) {
      int32 next_even_i = modulus * (i / modulus + 1);
      int32 min_i = state_info_[i - 1].backpointer,
          max_i = (next_even_i < num_states ?
                   state_info_[next_even_i].backpointer :
                   num_states - 1);

      double min_cost = std::numeric_limits<double>::infinity();
      int32 best_backpointer = -1;
      for (int32 k = min_i; k <= max_i; k++) {
        double inter_frame_cost = (k - i) * (k - i) * inter_frame_factor;
        double this_cost = prev_forward_cost(k) + inter_frame_cost;
        if (this_cost < min_cost) {
          min_cost = this_cost;
          best_backpointer = k;
        }
      }
      KALDI_ASSERT(best_backpointer != -1);
      state_info_[i].backpointer = best_backpointer;
      (*this_forward_cost)(i) = min_cost + local_cost(i);      
    }
    // This is a convenient time to set the pov_nccf field for all i.
    state_info_[i].pov_nccf = nccf_pov(i);
  }
    
}

void PitchFrameInfo::SetBestState(int32 best_state,
                                  std::vector<std::pair<int32, BaseFloat> >::iterator iter) {
  // This function would naturally be recursive, but we have coded this to avoid
  // recursion, which would otherwise eat up the stack.  Think of it as a static
  // member function, except we do use "this" right at the beginning.
  PitchFrameInfo *this_info = this;  // it will change in the loop.
  while (this_info != NULL) {
    PitchFrameInfo *prev_info = this_info->prev_info_;
    if (best_state == this_info->cur_best_state_) return;  // no change
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->first = best_state;
    size_t state_info_index = best_state - this_info->state_offset_;
    KALDI_ASSERT(state_info_index < this_info->state_info_.size());
    this_info->cur_best_state_ = best_state;
    best_state = this_info->state_info_[state_info_index].backpointer;
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->second = this_info->state_info_[state_info_index].pov_nccf;
    this_info = prev_info;
    iter--;
  }    
}


int32 PitchFrameInfo::ComputeLatency(int32 max_latency) {
  if (max_latency <= 0) return 0;

  int32 latency = 0;
  
  // This function would naturally be recursive, but we have coded this to avoid
  // recursion, which would otherwise eat up the stack.  Think of it as a static
  // member function, except we do use "this" right at the beginning.
  // This function is called only on the most recent PitchFrameInfo object.
  int32 num_states = state_info_.size();
  int32 min_living_state = 0, max_living_state = num_states - 1;
  PitchFrameInfo *this_info = this; // it will change in the loop.

  
  for (; this_info != NULL && latency < max_latency;) {
    int32 offset = this_info->state_offset_;
    KALDI_ASSERT(min_living_state >= offset &&
                 max_living_state - offset < this_info->state_info_.size());
    min_living_state =
        this_info->state_info_[min_living_state - offset].backpointer;
    max_living_state =
        this_info->state_info_[max_living_state - offset].backpointer;
    if (min_living_state == max_living_state) {
      return latency;
    }
    this_info = this_info->prev_info_;
    if (this_info != NULL)  // avoid incrementing latency for frame -1,
      latency++;            // as it's not a real frame.
  }
  return latency;
}

void PitchFrameInfo::Cleanup(PitchFrameInfo *prev_frame) {
  KALDI_ERR << "Cleanup not implemented.";
}





// We could inherit from OnlineBaseFeature as we have the same interface,
// but this will unnecessary force a lot of our functions to be virtual.
class OnlinePitchFeatureImpl {
 public:
  explicit OnlinePitchFeatureImpl(const PitchExtractionOptions &opts);

  int32 Dim() const { return 2; }

  int32 NumFramesReady() const;

  bool IsLastFrame(int32 frame) const;

  void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);

  void InputFinished();

  ~OnlinePitchFeatureImpl();


  // Copy-constructor, can be used to obtain a new copy of this object,
  // any state from this utterance.
  OnlinePitchFeatureImpl(const OnlinePitchFeatureImpl &other);
  
 private:

  /// This function works out from the signal how many frames are currently
  /// available to process (this is called from inside AcceptWaveform()).
  /// Note: the number of frames differs slightly from the number the
  /// old pitch code gave.
  int32 NumFramesAvailable(int64 num_downsampled_samples) const;

  /// This function extracts from the signal the samples numbered from
  /// "sample_index" (numbered in the full downsampled signal, not just this
  /// part), and of length equal to window->Dim().  It uses the data members
  /// downsampled_samples_discarded_ and downsampled_signal_remainder_, as well
  /// as the more recent part of the downsampled wave "downsampled_wave_part"
  /// which is provided.
  ///
  /// @param downsampled_wave_part  One chunk of the downsampled wave.
  /// @param sample_index  The desired starting sample index (measured from
  ///                      the start of the whole signal, not just this part).
  /// @param window  The part of the signal is output to here.
  void ExtractFrame(const VectorBase<BaseFloat> &downsampled_wave_part,
                    int64 frame_index,
                    VectorBase<BaseFloat> *window);

  /// This function updates downsampled_signal_remainder_,
  /// downsampled_samples_processed_, and signal_sumsq_; it's called at the end
  /// of AcceptWaveform().
  void UpdateRemainder(const VectorBase<BaseFloat> &downsampled_wave_part);

  
  // The following variables don't change throughout the lifetime
  // of this object.
  PitchExtractionOptions opts_;

  // the first lag of the downsampled signal at which we measure NCCF
  int32 nccf_first_lag_;
  // the last lag of the downsampled signal at which we measure NCCF
  int32 nccf_last_lag_;

  // The log-spaced lags at which we will resample the NCCF
  Vector<BaseFloat> lags_;
  
  // This object is used to resample from evenly spaced to log-evenly-spaced
  // nccf values.  It's a pointer for convenience of initialization, so we don't
  // have to use the initializer from the constructor.
  ArbitraryResample *nccf_resampler_;

  // The following objects may change during the lifetime of this object.

  // This object is used to resample the signal.
  LinearResample *signal_resampler_;
  
  // frame_info_ is indexed by [frame-index + 1].  frame_info_[0] is an object
  // that corresponds to frame -1, which is not a real frame.
  std::vector<PitchFrameInfo*> frame_info_;

  // Current number of frames which we can't output because Viterbi has not
  // converged for them, or opts_.max_frames_latency if we have reached that
  // limit.
  int32 frames_latency_;
  
  // The forward-cost at the current frame (the last frame in frame_info_);
  // this has the same dimension as lags_.
  Vector<double> forward_cost_;

  // The resampled-lag index and the NCCF (as computed for POV, without ballast
  // term) for each frame, as determined by Viterbi traceback from the best
  // final state.
  std::vector<std::pair<int32, BaseFloat> > lag_nccf_;
  
  bool input_finished_;

  /// sum-squared of previously processed parts of signal; used to get NCCF
  /// ballast term.  Denominator is downsampled_samples_processed_.
  double signal_sumsq_;
  
  /// downsampled_samples_processed is the number of samples (after
  /// downsampling) that we got in previous calls to AcceptWaveform().
  int64 downsampled_samples_processed_;
  /// This is a small remainder of the previous downsampled signal;
  /// it's used by ExtractFrame for frames near the boundary of two
  /// waveforms supplied to AcceptWaveform().
  Vector<BaseFloat> downsampled_signal_remainder_;
};


OnlinePitchFeatureImpl::OnlinePitchFeatureImpl(
    const PitchExtractionOptions &opts):
    opts_(opts), input_finished_(false), signal_sumsq_(0.0), 
    downsampled_samples_processed_(0) {

  signal_resampler_ = new LinearResample(opts.samp_freq, opts.resample_freq,
                                         opts.lowpass_cutoff,
                                         opts.lowpass_filter_width);
  
  double outer_min_lag = 1.0 / opts.max_f0 -
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / opts.min_f0 +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  nccf_first_lag_ = ceil(opts.resample_freq * outer_min_lag);
  nccf_last_lag_ = floor(opts.resample_freq * outer_max_lag);

  frames_latency_ = 0; // will be set in AcceptWaveform()
  
  // Choose the lags at which we resample the NCCF.
  SelectLags(opts, &lags_);

  // upsample_cutoff is the filter cutoff for upsampling the NCCF, which is the
  // Nyquist of the resampling frequency.  The NCCF is (almost completely)
  // bandlimited to around "lowpass_cutoff" (1000 by default), and when the
  // spectrum of this bandlimited signal is convolved with the spectrum of an
  // impulse train with frequency "resample_freq", which are separated by 4kHz,
  // we get energy at -5000,-3000, -1000...1000, 3000..5000, etc.  Filtering at
  // half the Nyquist (2000 by default) is sufficient to get only the first
  // repetition.
  BaseFloat upsample_cutoff = opts.resample_freq * 0.5;  


  Vector<BaseFloat> lags_offset(lags_);
  // lags_offset equals lags_ (which are the log-spaced lag values we want to
  // measure the NCCF at) with nccf_first_lag_ / opts.resample_freq subtracted
  // from each element, so we can treat the measured NCCF values as as starting
  // from sample zero in a signal that starts at the point start /
  // opts.resample_freq.  This is necessary because the ArbitraryResample code
  // assumes that the input signal starts from sample zero.
  lags_offset.Add(-nccf_first_lag_ / opts.resample_freq);

  int32 num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_;
  
  nccf_resampler_ = new ArbitraryResample(num_measured_lags, opts.resample_freq,
                                          upsample_cutoff, lags_offset,
                                          opts.upsample_filter_width);

  // add a PitchInfo object for frame -1 (not a real frame).
  frame_info_.push_back(new PitchFrameInfo(lags_.Dim()));
  // zeroes forward_cost_; this is what we want for the fake frame -1.
  forward_cost_.Resize(lags_.Dim()); 

}


int32 OnlinePitchFeatureImpl::NumFramesAvailable(
    int64 num_downsampled_samples) const {
  int32 frame_shift = opts_.NccfWindowShift(),
      frame_length = opts_.NccfWindowSize(),
      full_frame_length = frame_length + nccf_last_lag_;
  if (num_downsampled_samples < full_frame_length) return 0;
  else return ((num_downsampled_samples - full_frame_length) / frame_shift) + 1;
}

void OnlinePitchFeatureImpl::UpdateRemainder(
    const VectorBase<BaseFloat> &downsampled_wave_part) {
  // frame_info_ has an extra element at frame-1, so subtract
  // one from the length.
  int64 num_frames = static_cast<int64>(frame_info_.size()) - 1,
      next_frame = num_frames + 1,
      frame_shift = opts_.NccfWindowShift(),
      next_frame_sample = frame_shift * next_frame;
  
  signal_sumsq_ += VecVec(downsampled_wave_part, downsampled_wave_part);
  
  // next_frame_sample is the first sample index we'll need for the
  // next frame.
  int64 next_downsampled_samples_processed =
      downsampled_samples_processed_ + downsampled_wave_part.Dim();
  
  if (next_frame_sample > next_downsampled_samples_processed) {
    // this could only happen in the weird situation that the full frame length
    // is less than the frame shift.
    int32 full_frame_length = opts_.NccfWindowSize() + nccf_last_lag_;
    KALDI_ASSERT(full_frame_length < frame_shift && "Code error");
    downsampled_signal_remainder_.Resize(0);
  } else {
    Vector<BaseFloat> new_remainder(next_downsampled_samples_processed -
                                    next_frame_sample);
    // note: next_frame_sample is the index into the entire signal, of
    // new_remainder(0).
    // i is the absolute index of the signal.
    for (int64 i = next_frame_sample;
         i < next_downsampled_samples_processed; i++) {
      if (i >= downsampled_samples_processed_) {  // in current signal.
        new_remainder(i - next_frame_sample) =
            downsampled_wave_part(i - downsampled_samples_processed_);
      } else { // in old remainder; only reach here if waveform supplied is tiny.
        new_remainder(i - next_frame_sample) =
            downsampled_signal_remainder_(i - downsampled_samples_processed_ +
                                          downsampled_signal_remainder_.Dim());
      }
    }
    downsampled_signal_remainder_.Swap(&new_remainder);
  }
  downsampled_samples_processed_ = next_downsampled_samples_processed;
}

void OnlinePitchFeatureImpl::ExtractFrame(
    const VectorBase<BaseFloat> &downsampled_wave_part,
    int64 sample_index,
    VectorBase<BaseFloat> *window) {
  int32 full_frame_length = window->Dim();
  int32 offset = static_cast<int32>(sample_index -
                                    downsampled_samples_processed_);
  // "offset" is the offset of the start of the frame, into this
  // signal.
  if (offset >= 0) {
    // frame is full inside the new part of the signal.
    window->CopyFromVec(downsampled_wave_part.Range(offset, full_frame_length));
  } else {
    // frame is partly in the remainder and partly in the new part.
    int32 remainder_offset = downsampled_signal_remainder_.Dim() + offset;
    KALDI_ASSERT(remainder_offset > 0);  // or we didn't keep enough remainder.
    KALDI_ASSERT(offset + full_frame_length > 0);  // or we should have
                                                   // processed this frame last
                                                   // time.
    
    int32 old_length = -offset, new_length = offset + full_frame_length;
    window->Range(0, old_length).CopyFromVec(
        downsampled_signal_remainder_.Range(remainder_offset, old_length));
    window->Range(old_length, new_length).CopyFromVec(
        downsampled_wave_part.Range(0, new_length));
  }

  if (opts_.preemph_coeff != 0.0) {
    BaseFloat preemph_coeff = opts_.preemph_coeff;
    for (int32 i = window->Dim() - 1; i > 0; i--)
      (*window)(i) -= preemph_coeff * (*window)(i-1);
    (*window)(0) *= (1.0 - preemph_coeff);
  }
}

bool OnlinePitchFeatureImpl::IsLastFrame(int32 frame) const {
  int32 T = NumFramesReady();
  KALDI_ASSERT(frame < T);
  return (input_finished_ && frame + 1 == T);
}

int32 OnlinePitchFeatureImpl::NumFramesReady() const {
  int32 num_frames = lag_nccf_.size(),
      latency = frames_latency_;
  KALDI_ASSERT(latency <= num_frames);
  return num_frames - latency;
}


void OnlinePitchFeatureImpl::GetFrame(int32 frame,
                                      VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(frame < NumFramesReady() && feat->Dim() == 2);
  (*feat)(0) = lag_nccf_[frame].second;
  (*feat)(1) = 1.0 / lags_(lag_nccf_[frame].first);
}

void OnlinePitchFeatureImpl::InputFinished() {
  input_finished_ = true;
  frames_latency_ = 0;
  {
    int32 num_frames = NumFramesReady();
    KALDI_VLOG(3) << "Pitch-tracking Viterbi cost is "
                  << (forward_cost_.Min() / num_frames)
                  << " per frame, over " << num_frames << " frames.";
  }
}


OnlinePitchFeatureImpl::~OnlinePitchFeatureImpl() {
  delete nccf_resampler_;
  delete signal_resampler_;
  for (size_t i = 0; i < frame_info_.size(); i++)
    delete frame_info_[i];
}

void OnlinePitchFeatureImpl::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &wave) {
  // we never flush out the last few samples of input waveform... this would on
  // very rare occasions affect the number of frames processed, but since the
  // number of frames produced is anyway different from the MFCC/PLP processing
  // code, we already need to tolerate that.
  const bool flush = false;

  Vector<BaseFloat> downsampled_wave;
  signal_resampler_->Resample(wave, flush, &downsampled_wave);

  // these variables will be used to compute the root-mean-square value of the
  // signal for the ballast term.
  BaseFloat cur_sumsq = signal_sumsq_;
  int64 cur_num_frames = downsampled_samples_processed_,
      prev_frame_end_sample = 0;
  if (!opts_.nccf_ballast_online) {
    cur_sumsq += VecVec(downsampled_wave, downsampled_wave);
    cur_num_frames += downsampled_wave.Dim();
  }

  // num_frames is the total number of frames we can now process, including
  // previously processed ones.
  int32 num_frames = NumFramesAvailable(
      downsampled_samples_processed_ + downsampled_wave.Dim());
  // "frame" is the next frame-index we process
  int32 frame = frame_info_.size() - 1;

  int32 num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_,
      num_resampled_lags = lags_.Dim(),
      frame_shift = opts_.NccfWindowShift(),
      basic_frame_length = opts_.NccfWindowSize(),
      full_frame_length = basic_frame_length + nccf_last_lag_;
  
  Vector<BaseFloat> window(full_frame_length),
      inner_prod(num_measured_lags),
      norm_prod(num_measured_lags),
      nccf_pitch(num_measured_lags),
      nccf_pov(num_measured_lags),
      nccf_pitch_resampled(num_resampled_lags),
      nccf_pov_resampled(num_resampled_lags);
  Vector<double> cur_forward_cost(num_resampled_lags);

  // Because the resampling of the NCCF is more efficient when grouped together,
  // we first compute the NCCF for all frames, then resample as a matrix, then
  // do the Viterbi [that happens inside the constructor of PitchFrameInfo].
  
  for (; frame < num_frames; frame++) {
    // start_sample is index into the whole wave, not just this part.
    int64 start_sample = static_cast<int64>(frame) * frame_shift;
    ExtractFrame(downsampled_wave, start_sample, &window);
    if (opts_.nccf_ballast_online) {
      // use only up to end of current frame to compute root-mean-square value.
      // end_sample will be the sample-index into "downsampled_wave", so
      // not really comparable to start_sample.
      int64 end_sample = start_sample + full_frame_length -
          downsampled_samples_processed_;
      KALDI_ASSERT(end_sample > 0);  // or should have processed this frame last
                                     // time.  Note: end_sample is one past last
                                     // sample.
      SubVector<BaseFloat> new_part(downsampled_wave, prev_frame_end_sample,
                                    end_sample - prev_frame_end_sample);
      cur_num_frames += new_part.Dim();
      cur_sumsq += VecVec(new_part, new_part);
      prev_frame_end_sample = end_sample;
    }
    double mean_square = cur_sumsq / cur_num_frames;
    ComputeCorrelation(window, nccf_first_lag_, nccf_last_lag_,
                       basic_frame_length, &inner_prod, &norm_prod);
    double nccf_ballast_pov = 0.0,
        nccf_ballast_pitch = pow(mean_square * basic_frame_length, 2) *
                              opts_.nccf_ballast;
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pitch, &nccf_pitch);
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pov, &nccf_pov);

    if (frame == 100) {
      KALDI_VLOG(4) << "frame 100: nccf_pitch is " << nccf_pitch;
    }

    nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
    nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);


    if (frame == 100) {
      KALDI_VLOG(4) << "frame 100: nccf_pitch_resampled is " << nccf_pitch_resampled;
    }

    
    PitchFrameInfo *prev_info = frame_info_.back(),
        *cur_info = new PitchFrameInfo(opts_, nccf_pitch_resampled,
                                       nccf_pov_resampled, lags_,
                                       forward_cost_, prev_info,
                                       &cur_forward_cost);
    forward_cost_.Swap(&cur_forward_cost);
    frame_info_.push_back(cur_info);
  }
  UpdateRemainder(downsampled_wave);

  // Trace back the best-path.
  int32 best_final_state;
  forward_cost_.Min(&best_final_state);
  lag_nccf_.resize(frame_info_.size() - 1);  // will keep any existing data.
  std::vector<std::pair<int32, BaseFloat> >::iterator last_iter =
      lag_nccf_.end() - 1;
  frame_info_.back()->SetBestState(best_final_state, last_iter);
  frames_latency_ = frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
}



// Some functions that forward from OnlinePitchFeature to
// OnlinePitchFeatureImpl.
int32 OnlinePitchFeature::NumFramesReady() const {
  return impl_->NumFramesReady();
}

OnlinePitchFeature::OnlinePitchFeature(const PitchExtractionOptions &opts):
    impl_(new OnlinePitchFeatureImpl(opts)) { }

bool OnlinePitchFeature::IsLastFrame(int32 frame) const {
  return impl_->IsLastFrame(frame);
}

void OnlinePitchFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  impl_->GetFrame(frame, feat);
}

void OnlinePitchFeature::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  impl_->AcceptWaveform(sampling_rate, waveform);
}

void OnlinePitchFeature::InputFinished() {
  impl_->InputFinished();
}

OnlinePitchFeature::~OnlinePitchFeature() {
  delete impl_;
}
    

void ComputeKaldiPitchNew(const PitchExtractionOptions &opts,
                          const VectorBase<BaseFloat> &wave,
                          Matrix<BaseFloat> *output) {
  OnlinePitchFeature pitch_extractor(opts);
  pitch_extractor.AcceptWaveform(opts.samp_freq, wave);
  pitch_extractor.InputFinished();
  int32 num_frames = pitch_extractor.NumFramesReady();
  if (num_frames == 0) {
    KALDI_WARN << "No frames output in pitch extraction";
    output->Resize(0, 0);
    return;
  }
  output->Resize(num_frames, 2);
  for (int32 frame = 0; frame < num_frames; frame++) {
    SubVector<BaseFloat> row(*output, frame);
    pitch_extractor.GetFrame(frame, &row);
  }
}




}  // namespace kaldi

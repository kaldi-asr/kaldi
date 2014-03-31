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
#include "feat/mel-computations.h"

namespace kaldi {


/*
  WeightedMovingWindowNormalize is doing weighted moving window normalization.

  The simple moving window normalization would be to set
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
  if (ndash > 1.0) ndash = 1.0; // just in case it was slightly outside [-1, 1]

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
  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  if (static_cast<int32>(nsamp) < frame_length)
    return 0;
  else
    return (1 + ((nsamp - frame_length) / frame_shift));
}
void PreemphasizeFrame(VectorBase<double> *waveform, double preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  KALDI_ASSERT(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
  for (int32 i = waveform->Dim()-1; i > 0; i--)
    (*waveform)(i) -= preemph_coeff * (*waveform)(i-1);
  (*waveform)(0) -= preemph_coeff * (*waveform)(0);
}

void ExtractFrame(const VectorBase<double> &wave,
                  int32 frame_num,
                  const PitchExtractionOptions &opts,
                  Vector<double> *window) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  int32 outer_max_lag = round(opts.resample_freq / opts.min_f0) +
      round(opts.lowpass_filter_width/2);

  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  int32 start = frame_shift * frame_num;
  int32 end = start + outer_max_lag  + frame_length;
  int32 frame_length_new = frame_length + outer_max_lag;

  if (window->Dim() != frame_length_new)
    window->Resize(frame_length_new);

  SubVector<double> wave_part(wave, start,
                              std::min(frame_length_new, wave.Dim()-start));

  SubVector<double>  window_part(*window, 0,
                                 std::min(frame_length_new, wave.Dim()-start));
  window_part.CopyFromVec(wave_part);

  if (opts.preemph_coeff != 0.0)
    PreemphasizeFrame(&window_part, opts.preemph_coeff);

  if (end > wave.Dim())
    SubVector<double>(*window, frame_length_new-end+wave.Dim(),
                      end-wave.Dim()).SetZero();
}

// A special case of ArbitraryResample, when resample point is linear (which
// is the most commonly used). Provided by Yanqing Sun & Junjie Wang, and
// could generates exactly the same results as ArbitraryResample. The benefits
// are to get rid of num_samples, to save calculation and also could be used
// on-line. Thanks Dan for providing the algorithm. The key is to work out the
// smallest number of input samples and output sample periods that give the
// exact same time.
class LinearResample {
 public:
     LinearResample(double samp_rate_in,
                    double samp_rate_out,
                    double filter_cutoff,
                    int32 num_zeros_upsample):
      samp_rate_in_(samp_rate_in),
      samp_rate_out_(samp_rate_out),
      filter_cutoff_(filter_cutoff),
      num_zeros_upsample_(num_zeros_upsample) {
    KALDI_ASSERT(samp_rate_in > 0.0 &&
                 samp_rate_out > 0.0 && samp_rate_in != samp_rate_out &&
                 filter_cutoff > 0.0 &&
                 filter_cutoff * 2.0 <= samp_rate_in
                 && num_zeros_upsample > 0);
    // pre-set some parameters, set up weights_ and indices_.
    PreSet();
    SetIndex();
    SetWeights();
  }

  void Resample(const VectorBase<double> &input,
                VectorBase<double> *output) {
    // each row of "input" corresponds to the data to resample;
    // the corresponding row of "output" is the resampled data.
    int32 num_samples_in_ = input.Dim();
    int32 resampled_len = 1 + static_cast<int>(num_samples_in_ / frame_shift_);
    if (output->Dim() != resampled_len) resampled_len = output->Dim();

    for (int32 i = 0; i < resampled_len; i++) {
      int32 inner_i = i % num_weights_;  // the index of weight to be used
      int32 offset = (i - inner_i) * frame_shift_;
      int32 fake_first_index = indexes_[inner_i].first_index + offset,
            fake_last_index = indexes_[inner_i].last_index + offset;
      int32 first_index = std::max(0, fake_first_index),
            last_index = std::min((num_samples_in_ - 1), fake_last_index);
      int32 num_indices = last_index - first_index + 1;
      SubVector<double> input_part(input, first_index, num_indices);
      Vector<double> &weights = weights_[inner_i];
      if (num_indices == weights.Dim())
        (*output)(i) = VecVec(input_part, weights);
      else if (fake_first_index >= 0) {
        SubVector<double> weight_vec(weights, 0, num_indices);
        (*output)(i) = VecVec(input_part, weight_vec);
      } else if (fake_first_index < 0) {
        SubVector<double> weight_vec(weights, -fake_first_index, num_indices);
        (*output)(i) = VecVec(input_part, weight_vec);
      }
    }
    output->Scale(1.0/samp_rate_in_);
  }
 private:
  void PreSet() {
    int32 samp_rate_gcd = Gcd(static_cast<int>(samp_rate_in_),
        static_cast<int>(samp_rate_out_));
    num_weights_ = static_cast<int>(samp_rate_out_) / samp_rate_gcd;
    half_frame_length_ = samp_rate_in_ *
        num_zeros_upsample_ / (2.0 * filter_cutoff_);
    frame_shift_ = samp_rate_in_ / samp_rate_out_;
  }
  void SetIndex() {
    indexes_.resize(num_weights_);
    for (int32 i = 0; i < num_weights_; i++) {
      indexes_[i].first_index =
        static_cast<int>(ceil(i*frame_shift_ - half_frame_length_));
      indexes_[i].last_index =
        static_cast<int>(floor(i*frame_shift_ + half_frame_length_) + 1);
      indexes_[i].num_indices =
        indexes_[i].last_index - indexes_[i].first_index + 1;
    }
  }
  void SetWeights() {
    weights_.resize(num_weights_);
    double t, j_double, tn;
    for (int32 i = 0; i < num_weights_; i++) {
      int32 first_index = indexes_[i].first_index;
      int32 num_indices = indexes_[i].num_indices;
      weights_[i].Resize(num_indices);
      for (int32 j = 0; j < num_indices; j++) {
        j_double = static_cast<double>(j);
        t = i / samp_rate_out_ - (j_double + first_index) / samp_rate_in_;
        tn = FilterFunc(t);
        weights_[i](j) = tn;
      }
    }
  }
  double FilterFunc(const double &t) {
    double f_t = 0,  win = 0;

    if (fabs(t) < num_zeros_upsample_ /(2.0 * filter_cutoff_))
      win = 0.5 * (1 + cos(M_2PI * filter_cutoff_ / num_zeros_upsample_ * t));
    if (t != 0)
      f_t = sin(M_2PI * filter_cutoff_ * t) / (M_PI * t) * win;
    else
      f_t = 2 * filter_cutoff_ * win;
    return f_t;
  }
  double samp_rate_in_;
  double samp_rate_out_;
  double filter_cutoff_;
  int32 num_zeros_upsample_;
  int32 num_weights_;   // minimal number of weights
  double half_frame_length_;
  double frame_shift_;  // or called 'dt'
  struct IndexInfo {
    // These infos are just theoretical and relative values, ignoring
    // boundaries and offsets. Will be modified in 'Upsample'.
    int32 first_index;  // The first input-sample index that we sum
                        // over, for this output-sample index.
    int32 last_index;   // The last input-sample index that we sum
                        // over, for this output-sample index.
    int32 num_indices;  // The number of indices that we sum over.
  };
  // both indexes_.size() and weights_.size() equal num_weights_.
  std::vector<IndexInfo> indexes_;
  std::vector<Vector<double> > weights_;
};

class ArbitraryResample {
 public:
  ArbitraryResample(int32 num_samples_in, double samp_rate_in,
                    double filter_cutoff,
                    const std::vector<double> &sample_points,
                    int32 num_zeros_upsample):
      num_samples_in_(num_samples_in),
      samp_rate_in_(samp_rate_in),
      filter_cutoff_(filter_cutoff),
      num_zeros_upsample_(num_zeros_upsample) {
    KALDI_ASSERT(num_samples_in > 0 && samp_rate_in > 0.0 &&
                 filter_cutoff > 0.0 &&
                 filter_cutoff * 2.0 <= samp_rate_in
                 && num_zeros_upsample > 0);
    // set up weights_ and indices_.  Please try to keep all functions short and
    SetIndex(sample_points);
    SetWeights(sample_points);
  }

  int32 NumSamplesIn() const { return num_samples_in_; }
  int32 NumSamplesOut() const { return indexes_.size(); }

  void Resample(const MatrixBase<double> &input,
                MatrixBase<double> *output) {
    // each row of "input" corresponds to the data to resample;
    // the corresponding row of "output" is the resampled data.

    KALDI_ASSERT(input.NumRows() == output->NumRows() &&
                 input.NumCols() == num_samples_in_ &&
                 output->NumCols() == indexes_.size());

    Vector<double> output_col(output->NumRows());
    for (int32 i = 0; i < NumSamplesOut(); i++) {
      SubMatrix<double> input_part(input, 0, input.NumRows(),
                                   indexes_[i].first_index,
                                   indexes_[i].num_indices);
      const Vector<double> &weight_vec(weights_[i]);
      output_col.AddMatVec(1.0/samp_rate_in_, input_part,
                           kNoTrans, weight_vec, 0.0);
      output->CopyColFromVec(output_col, i);
    }
  }
 private:
  void SetIndex(const std::vector<double> &sample_points) {
    int32 last_ind, num_sample = sample_points.size();
    indexes_.resize(num_sample);
    for (int32  i = 0; i < num_sample; i++) {
      indexes_[i].first_index = std::max(0,
        static_cast<int>(ceil(samp_rate_in_ * (sample_points[i]
        - num_zeros_upsample_/(2.0 * filter_cutoff_)))));
      last_ind = std::min((num_samples_in_ - 1),
        static_cast<int>(floor(samp_rate_in_ *
        (sample_points[i] + num_zeros_upsample_ / (2.0 * filter_cutoff_)))
        + 1));
      indexes_[i].num_indices = last_ind - indexes_[i].first_index + 1;
    }
  }

  void SetWeights(const std::vector<double> &sample_points) {
    int32 num_samples_out = NumSamplesOut();
    weights_.resize(num_samples_out);
    double t, j_double, tn;
    for (int32 i = 0; i < num_samples_out; i++) {
      weights_[i].Resize(indexes_[i].num_indices);
      for (int32 j = 0 ; j < indexes_[i].num_indices; j++) {
        j_double = static_cast<double>(j);
        t = sample_points[i]
          - (j_double + indexes_[i].first_index) / samp_rate_in_;
        tn = FilterFunc(t);
        weights_[i](j) = tn;
      }
    }
  }
  double FilterFunc(const double &t) {
    double f_t = 0,  win = 0;

    if (fabs(t) < num_zeros_upsample_ /(2.0 * filter_cutoff_))
      win = 0.5 * (1 + cos(M_2PI * filter_cutoff_ / num_zeros_upsample_ * t));
    if (t != 0)
      f_t = sin(M_2PI * filter_cutoff_ * t) / (M_PI * t) * win;
    else
      f_t = 2 * filter_cutoff_ * win;
    return f_t;
  }
  int32 num_samples_in_;
  double samp_rate_in_;
  double filter_cutoff_;
  int32 num_zeros_upsample_;
  struct IndexInfo {
    int32 first_index;  // The first input-sample index that we sum
                        // over, for this output-sample index.
    int32 num_indices;  // The number of indices that we sum over.
  };
  // both indexes_.size() weights_.size() equal sample_points.size().
  std::vector<IndexInfo> indexes_;
  std::vector<Vector<double> > weights_;
};


void PreProcess(const PitchExtractionOptions opts,
                const Vector<double> &wave,
                Vector<double> *processed_wave) {
  KALDI_ASSERT(processed_wave != NULL);
  // down-sample and Low-Pass filtering the input wave
  int32 num_samples_in = wave.Dim();
  double dt = opts.samp_freq / opts.resample_freq;
  int32 resampled_len = 1 + static_cast<int>(num_samples_in / dt);
  processed_wave->Resize(resampled_len);  // filtered wave
  LinearResample resample(opts.samp_freq, opts.resample_freq,
                          opts.lowpass_cutoff,
                          opts.lowpass_filter_width);
  resample.Resample(wave, processed_wave);

  // Normalize input signal using rms
  double rms = pow(VecVec((*processed_wave), (*processed_wave))
    / processed_wave->Dim(), 0.5);
  if (rms != 0.0)
    (*processed_wave).Scale(1.0 / rms);
}

void Nccf(const Vector<double> &wave,
          int32 start, int32 end,
          int32 nccf_window_size,
          Vector<double> *inner_prod,
          Vector<double> *norm_prod) {
  Vector<double> zero_mean_wave(wave);
  SubVector<double> wave_part(wave, 0, nccf_window_size);
  // subtract mean-frame from wave
  zero_mean_wave.Add(-wave_part.Sum() / nccf_window_size);
  double e1, e2, sum;
  SubVector<double> sub_vec1(zero_mean_wave, 0, nccf_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  for (int32 lag = start; lag < end; lag++) {
    SubVector<double> sub_vec2(zero_mean_wave, lag, nccf_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*inner_prod)(lag-start) = sum;
    (*norm_prod)(lag-start) = e1 * e2;
  }
}

void ProcessNccf(const Vector<double> &inner_prod,
                 const Vector<double> &norm_prod,
                 const double &a_fact,
                 int32 start, int32 end,
                 SubVector<double> *autocorr) {
  for (int32 lag = start; lag < end; lag++) {
    if (norm_prod(lag-start) != 0.0)
      (*autocorr)(lag) =
        inner_prod(lag-start) / pow(norm_prod(lag-start) + a_fact, 0.5);
    KALDI_ASSERT((*autocorr)(lag) < 1.01 && (*autocorr)(lag) > -1.01);
  }
}

void SelectLag(const PitchExtractionOptions &opts,
               int32 *state_num,
               Vector<double> *lags) {
  // choose lags relative to acceptable pitch tolerance
  double min_lag = 1.0 / (1.0 * opts.max_f0),
      max_lag = 1.0 / (1.0 * opts.min_f0);
  double delta_lag = opts.upsample_filter_width/(2.0 * opts.resample_freq);

  int32 lag_size = 1 + round(log((max_lag + delta_lag) / (min_lag - delta_lag))
      / log(1.0 + opts.delta_pitch));
  lags->Resize(lag_size);

  // we choose sequence of lags which leads to delta_pitch difference in
  // pitch_space.
  double lag = min_lag;
  int32 count = 0;
  while (lag <= max_lag) {
    (*lags)(count) = lag;
    count++;
    lag = lag * (1 + opts.delta_pitch);
  }
  lags->Resize(count, kCopyData);
  (*state_num) = count;
}

class PitchExtractor {
 public:
  explicit PitchExtractor(const PitchExtractionOptions &opts,
                          const Vector<double> lags,
                          int32 state_num,
                          int32 num_frames) :
      opts_(opts),
      state_num_(state_num),
      num_frames_(num_frames),
      lags_(lags) {
    frames_.resize(num_frames_+1);
    for (int32 i = 0; i < num_frames_+1; i++) {
      frames_[i].local_cost.Resize(state_num_);
      frames_[i].obj_func.Resize(state_num_);
      frames_[i].back_pointers.Resize(state_num_);
    }
  }
  ~PitchExtractor() {}

  void ComputeLocalCost(const Matrix<double> &autocorrelation) {
    Vector<double> correl(state_num_);

    for (int32 i = 1; i < num_frames_ + 1; i++) {
      SubVector<double> frame(autocorrelation.Row(i-1));
      Vector<double> local_cost(state_num_);
      for (int32 j = 0; j < state_num_; j++)
        correl(j) = frame(j);
      // compute the local cost
      frames_[i].local_cost.Add(1.0);
      frames_[i].local_cost.AddVec(-1.0, correl);
      Vector<double> corr_lag_cost(state_num_);
      corr_lag_cost.AddVecVec(opts_.soft_min_f0, correl, lags_, 0);
      frames_[i].local_cost.AddVec(1.0, corr_lag_cost);
    }  // end of loop over frames
  }
  void FastViterbi(const Matrix<double> &correl) {
    ComputeLocalCost(correl);
    double intercost, min_c, this_c;
    int best_b, min_i, max_i;
    BaseFloat delta_pitch_sq = log(1 + opts_.delta_pitch)
      * log(1 + opts_.delta_pitch);
    // loop over frames
    for (int32 t = 1; t < num_frames_ + 1; t++) {
      // Forward Pass
      for (int32 i = 0; i < state_num_; i++) {
        if ( i == 0 )
          min_i = 0;
        else
          min_i = frames_[t].back_pointers(i-1);
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
        frames_[t].back_pointers(i) = best_b;
        frames_[t].obj_func(i) = min_c + frames_[t].local_cost(i);
      }
      // Backward Pass
      for (int32 i = state_num_-1; i >= 0; i--) {
        if (i == state_num_-1)
          max_i = state_num_-1;
        else
          max_i = frames_[t].back_pointers(i+1);
        min_c = frames_[t].obj_func(i) - frames_[t].local_cost(i);
        best_b = frames_[t].back_pointers(i);

        for (int32 k = i+1 ; k <= max_i; k++) {
          intercost = (i-k) * (i-k) * delta_pitch_sq;
          this_c = frames_[t-1].obj_func(k)+ opts_.penalty_factor *intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].back_pointers(i) = best_b;
        frames_[t].obj_func(i) = min_c + frames_[t].local_cost(i);
      }
    }
    // FindBestPath(resampled_nccf_pov);
  }

  void FindBestPath(const Matrix<double> &correlation) {
    // Find the Best path using backpointers
    int32 i = num_frames_;
    int32 best;
    double l_opt;
    frames_[i].obj_func.Min(&best);
    while (i > 0) {
      l_opt = lags_(best);
      frames_[i].truepitch = 1.0 / l_opt;
      frames_[i].pov = correlation(i-1, best);
      best = frames_[i].back_pointers(best);
      i--;
    }
  }
  void GetPitch(Matrix<BaseFloat> *output) {
    output->Resize(num_frames_, 2);
    for (int32 frm = 0; frm < num_frames_; frm++) {
      (*output)(frm, 0) = static_cast<BaseFloat>(frames_[frm + 1].pov);
      (*output)(frm, 1) = static_cast<BaseFloat>(frames_[frm + 1].truepitch);
    }
  }
 private:
  PitchExtractionOptions opts_;
  int32 state_num_;      // number of states in Viterbi Computation
  int32 num_frames_;     // number of frames in input wave
  Vector<double> lags_;    // all lags used in viterbi
  struct PitchFrame {
    Vector<double> local_cost;
    Vector<double> obj_func;      // optimal objective function for frame i
    Vector<double> back_pointers;
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
  Vector<double> wave2(wave);

  // Preprocess the wave
  Vector<double> processed_wave(wave2.Dim());
  PreProcess(opts, wave2, &processed_wave);
  int32 num_states, rows_out = PitchNumFrames(processed_wave.Dim(), opts);
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is "
      << processed_wave.Dim() << ")";
  Vector<double> window;       // windowed waveform.
  double outer_min_lag = 1.0 / (1.0 * opts.max_f0) -
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / (1.0 * opts.min_f0) +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  int32 num_max_lag = round(outer_max_lag * opts.resample_freq) + 1;
  int32 num_lags = round(opts.resample_freq * outer_max_lag) -
      round(opts.resample_freq *  outer_min_lag) + 1;
  int32 start = round(opts.resample_freq  * outer_min_lag),
      end = round(opts.resample_freq / opts.min_f0) +
      round(opts.lowpass_filter_width / 2);

  Vector<double> lags;
  SelectLag(opts, &num_states, &lags);
  double a_fact_pitch = pow(opts.NccfWindowSize(), 4) * opts.nccf_ballast,
    a_fact_pov = pow(10, -9);
  Matrix<double> nccf_pitch(rows_out, num_max_lag + 1),
      nccf_pov(rows_out, num_max_lag + 1);
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    ExtractFrame(processed_wave, r, opts, &window);
    // compute nccf for pitch extraction
    Vector<double> inner_prod(num_lags), norm_prod(num_lags);
    Nccf(window, start, end, opts.NccfWindowSize(),
         &inner_prod, &norm_prod);
    SubVector<double> nccf_pitch_vec(nccf_pitch.Row(r));
    ProcessNccf(inner_prod, norm_prod, a_fact_pitch,
        start, end, &(nccf_pitch_vec));
    // compute the Nccf for Probability of voicing estimation
    SubVector<double> nccf_pov_vec(nccf_pov.Row(r));
    ProcessNccf(inner_prod, norm_prod, a_fact_pov,
        start, end, &(nccf_pov_vec));
  }
  std::vector<double> lag_vec(num_states);
  for (int32 i = 0; i < num_states; i++)
    lag_vec[i] = static_cast<double>(lags(i));
  // upsample the nccf to have better pitch and pov estimation

  // upsample_cutoff is the filter cutoff for upsampling the NCCF, which is the
  // Nyquist of the resampling frequency.  The NCCF is (almost completely)
  // bandlimited to around "lowpass_cutoff" (1000 by default), and when the
  // spectrum of this bandlimited signal is convolved with the spectrum of an
  // impulse train with frequency "resample_freq", which are separated by 4kHz,
  // we get energy at -5000,-3000, -1000...1000, 3000..5000, etc.  Filtering at
  // half the Nyquist (2000 by default) is sufficient to get only the first
  // repetition.
  BaseFloat upsample_cutoff = opts.resample_freq * 0.5;
  ArbitraryResample resample(num_max_lag + 1, opts.resample_freq,
                             upsample_cutoff,
                             lag_vec, opts.upsample_filter_width);
  Matrix<double> resampled_nccf_pitch(rows_out, num_states);
  resample.Resample(nccf_pitch, &resampled_nccf_pitch);
  Matrix<double> resampled_nccf_pov(rows_out, num_states);
  resample.Resample(nccf_pov, &resampled_nccf_pov);

  PitchExtractor pitch(opts, lags, num_states, rows_out);
  pitch.FastViterbi(resampled_nccf_pitch);
  pitch.FindBestPath(resampled_nccf_pov);
  output->Resize(rows_out, 2);  // (pov, pitch)
  pitch.GetPitch(output);
}

/**
   This function applies to the pitch the normal delta (time-derivative)
   computation using a five frame window, multiplying by a normalized version of
   the scales [ -2, 1, 0, 1, 2 ].  It then adds a small amount of noise to the
   output, in order to avoid peaks appearing in the distribution of delta pitch,
   that correspond to the discretization interval for log-pitch.
*/
void ExtractDeltaPitch(const PostProcessPitchOptions &opts,
                       const Vector<BaseFloat> &input,
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
                      const Matrix<BaseFloat> &input,
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


}  // namespace kaldi

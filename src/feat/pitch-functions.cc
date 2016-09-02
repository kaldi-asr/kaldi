// feat/pitch-functions.cc

// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer
//                    Xin Lei

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

#include <algorithm>
#include <limits>

#include "feat/feature-functions.h"
#include "feat/mel-computations.h"
#include "feat/online-feature.h"
#include "feat/pitch-functions.h"
#include "feat/resample.h"
#include "matrix/matrix-functions.h"

namespace kaldi {

/**
   This function processes the NCCF n to a POV feature f by applying the formula
     f = (1.0001 - n)^0.15  - 1.0
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

  BaseFloat r = -5.2 + 5.4 * Exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
                2.0 * Exp(-10.0 * ndash) + 4.2 * Exp(20.0 * (ndash - 1.0));
  // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
  BaseFloat p = 1.0 / (1 + Exp(-1.0 * r));
  KALDI_ASSERT(p - p == 0);  // Check for NaN/inf
  return p;
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
                 BaseFloat nccf_ballast,
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

  /// This function may be called for the last (most recent) PitchFrameInfo
  /// object with the best state (obtained from the externally held
  /// forward-costs). It traces back as far as needed to set the
  /// cur_best_state_, and as it's going it sets the lag-index and pov_nccf in
  /// pitch_pov_iter, which when it's called is an iterator to where to put the
  /// info for the final state; the iterator will be decremented inside this
  /// function.
  void SetBestState(int32 best_state,
      std::vector<std::pair<int32, BaseFloat> > &lag_nccf);

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
  explicit PitchFrameInfo(int32 num_states);

  /// This constructor is used for subsequent frames (not -1).
  PitchFrameInfo(PitchFrameInfo *prev);

  /// Record the nccf_pov value.
  ///  @param  nccf_pov     The nccf as computed for the POV computation (without ballast).
  void SetNccfPov(const VectorBase<BaseFloat> &nccf_pov);

  /// This constructor is used for frames apart from frame -1; the bulk of
  /// the Viterbi computation takes place inside this constructor.
  ///  @param  opts         The options as provided by the user
  ///  @param  nccf_pitch   The nccf as computed for the pitch computation
  ///                       (with ballast).
  ///  @param  nccf_pov     The nccf as computed for the POV computation
  ///                       (without ballast).
  ///  @param  lags         The log-spaced lags at which nccf_pitch and
  ///                       nccf_pov are sampled.
  ///  @param  prev_frame_forward_cost   The forward-cost vector for the
  ///                       previous frame.
  ///  @param  index_info   A pointer to a temporary vector used by this function
  ///  @param  this_forward_cost   The forward-cost vector for this frame
  ///                       (to be computed).
  void ComputeBacktraces(const PitchExtractionOptions &opts,
                         const VectorBase<BaseFloat> &nccf_pitch,
                         const VectorBase<BaseFloat> &lags,
                         const VectorBase<BaseFloat> &prev_forward_cost,
                         std::vector<std::pair<int32, int32> > *index_info,
                         VectorBase<BaseFloat> *this_forward_cost);
 private:
  // struct StateInfo is the information we keep for a single one of the
  // log-spaced lags, for a single frame.  This is a state in the Viterbi
  // computation.
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
PitchFrameInfo::PitchFrameInfo(int32 num_states)
    :state_info_(num_states), state_offset_(0),
    cur_best_state_(-1), prev_info_(NULL) { }


bool pitch_use_naive_search = false;  // This is used in unit-tests.


PitchFrameInfo::PitchFrameInfo(PitchFrameInfo *prev_info):
    state_info_(prev_info->state_info_.size()), state_offset_(0),
    cur_best_state_(-1), prev_info_(prev_info) { }

void PitchFrameInfo::SetNccfPov(const VectorBase<BaseFloat> &nccf_pov) {
  int32 num_states = nccf_pov.Dim();
  KALDI_ASSERT(num_states == state_info_.size());
  for (int32 i = 0; i < num_states; i++)
    state_info_[i].pov_nccf = nccf_pov(i);
}

void PitchFrameInfo::ComputeBacktraces(
    const PitchExtractionOptions &opts,
    const VectorBase<BaseFloat> &nccf_pitch,
    const VectorBase<BaseFloat> &lags,
    const VectorBase<BaseFloat> &prev_forward_cost_vec,
    std::vector<std::pair<int32, int32> > *index_info,
    VectorBase<BaseFloat> *this_forward_cost_vec) {
  int32 num_states = nccf_pitch.Dim();

  Vector<BaseFloat> local_cost(num_states, kUndefined);
  ComputeLocalCost(nccf_pitch, lags, opts, &local_cost);

  const BaseFloat delta_pitch_sq = pow(Log(1.0 + opts.delta_pitch), 2.0),
      inter_frame_factor = delta_pitch_sq * opts.penalty_factor;

  // index local_cost, prev_forward_cost and this_forward_cost using raw pointer
  // indexing not operator (), since this is the very inner loop and a lot of
  // time is taken here.
  const BaseFloat *prev_forward_cost = prev_forward_cost_vec.Data();
  BaseFloat *this_forward_cost = this_forward_cost_vec->Data();

  if (index_info->empty())
    index_info->resize(num_states);

  // make it a reference for more concise indexing.
  std::vector<std::pair<int32, int32> > &bounds = *index_info;

  /* bounds[i].first will be a lower bound on the backpointer for state i,
     bounds[i].second will be an upper bound on it.  We progressively tighten
     these bounds till we know the backpointers exactly.
  */

  if (pitch_use_naive_search) {
    // This branch is only taken in unit-testing code.
    for (int32 i = 0; i < num_states; i++) {
      BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
      int32 best_j = -1;
      for (int32 j = 0; j < num_states; j++) {
        BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
            + prev_forward_cost[j];
        if (this_cost < best_cost) {
          best_cost = this_cost;
          best_j = j;
        }
      }
      this_forward_cost[i] = best_cost;
      state_info_[i].backpointer = best_j;
    }
  } else {
    int32 last_backpointer = 0;
    for (int32 i = 0; i < num_states; i++) {
      int32 start_j = last_backpointer;
      BaseFloat best_cost = (start_j - i) * (start_j - i) * inter_frame_factor
          + prev_forward_cost[start_j];
      int32 best_j = start_j;

      for (int32 j = start_j + 1; j < num_states; j++) {
        BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
            + prev_forward_cost[j];
        if (this_cost < best_cost) {
          best_cost = this_cost;
          best_j = j;
        } else {  // as soon as the costs stop improving, we stop searching.
          break;  // this is a loose lower bound we're getting.
        }
      }
      state_info_[i].backpointer = best_j;
      this_forward_cost[i] = best_cost;
      bounds[i].first = best_j;  // this is now a lower bound on the
                                 // backpointer.
      bounds[i].second = num_states - 1;  // we have no meaningful upper bound
                                          // yet.
      last_backpointer = best_j;
    }

    // We iterate, progressively refining the upper and lower bounds until they
    // meet and we know that the resulting backtraces are optimal.  Each
    // iteration takes time linear in num_states.  We won't normally iterate as
    // far as num_states; normally we only do two iterations; when printing out
    // the number of iterations, it's rarely more than that (once I saw seven
    // iterations).  Anyway, this part of the computation does not dominate.
    for (int32 iter = 0; iter < num_states; iter++) {
      bool changed = false;
      if (iter % 2 == 0) {  // go backwards through the states
        last_backpointer = num_states - 1;
        for (int32 i = num_states - 1; i >= 0; i--) {
          int32 lower_bound = bounds[i].first,
              upper_bound = std::min(last_backpointer, bounds[i].second);
          if (upper_bound == lower_bound) {
            last_backpointer = lower_bound;
            continue;
          }
          BaseFloat best_cost = this_forward_cost[i];
          int32 best_j = state_info_[i].backpointer, initial_best_j = best_j;

          if (best_j == upper_bound) {
            // if best_j already equals upper bound, don't bother tightening the
            // upper bound, we'll tighten the lower bound when the time comes.
            last_backpointer = best_j;
            continue;
          }
          // Below, we have j > lower_bound + 1 because we know we've already
          // evaluated lower_bound and lower_bound + 1 [via knowledge of
          // this algorithm.]
          for (int32 j = upper_bound; j > lower_bound + 1; j--) {
            BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
                + prev_forward_cost[j];
            if (this_cost < best_cost) {
              best_cost = this_cost;
              best_j = j;
            } else {  // as soon as the costs stop improving, we stop searching,
              // unless the best j is still lower than j, in which case
              // we obviously need to keep moving.
              if (best_j > j)
                break;  // this is a loose lower bound we're getting.
            }
          }
          // our "best_j" is now an upper bound on the backpointer.
          bounds[i].second = best_j;
          if (best_j != initial_best_j) {
            this_forward_cost[i] = best_cost;
            state_info_[i].backpointer = best_j;
            changed = true;
          }
          last_backpointer = best_j;
        }
      } else {  // go forwards through the states.
        last_backpointer = 0;
        for (int32 i = 0; i < num_states; i++) {
          int32 lower_bound = std::max(last_backpointer, bounds[i].first),
              upper_bound = bounds[i].second;
          if (upper_bound == lower_bound) {
            last_backpointer = lower_bound;
            continue;
          }
          BaseFloat best_cost = this_forward_cost[i];
          int32 best_j = state_info_[i].backpointer, initial_best_j = best_j;

          if (best_j == lower_bound) {
            // if best_j already equals lower bound, we don't bother tightening
            // the lower bound, we'll tighten the upper bound when the time
            // comes.
            last_backpointer = best_j;
            continue;
          }
          // Below, we have j < upper_bound because we know we've already
          // evaluated that point.
          for (int32 j = lower_bound; j < upper_bound - 1; j++) {
            BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
                + prev_forward_cost[j];
            if (this_cost < best_cost) {
              best_cost = this_cost;
              best_j = j;
            } else {  // as soon as the costs stop improving, we stop searching,
              // unless the best j is still higher than j, in which case
              // we obviously need to keep moving.
              if (best_j < j)
                break;  // this is a loose lower bound we're getting.
            }
          }
          // our "best_j" is now a lower bound on the backpointer.
          bounds[i].first = best_j;
          if (best_j != initial_best_j) {
            this_forward_cost[i] = best_cost;
            state_info_[i].backpointer = best_j;
            changed = true;
          }
          last_backpointer = best_j;
        }
      }
      if (!changed)
        break;
    }
  }
  // The next statement is needed due to RecomputeBacktraces: we have to
  // invalidate the previously computed best-state info.
  cur_best_state_ = -1;
  this_forward_cost_vec->AddVec(1.0, local_cost);
}

void PitchFrameInfo::SetBestState(
    int32 best_state,
    std::vector<std::pair<int32, BaseFloat> > &lag_nccf) {

  // This function would naturally be recursive, but we have coded this to avoid
  // recursion, which would otherwise eat up the stack.  Think of it as a static
  // member function, except we do use "this" right at the beginning.

  std::vector<std::pair<int32, BaseFloat> >::reverse_iterator iter = lag_nccf.rbegin();

  PitchFrameInfo *this_info = this;  // it will change in the loop.
  while (this_info != NULL) {
    PitchFrameInfo *prev_info = this_info->prev_info_;
    if (best_state == this_info->cur_best_state_)
      return;  // no change
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->first = best_state;
    size_t state_info_index = best_state - this_info->state_offset_;
    KALDI_ASSERT(state_info_index < this_info->state_info_.size());
    this_info->cur_best_state_ = best_state;
    best_state = this_info->state_info_[state_info_index].backpointer;
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->second = this_info->state_info_[state_info_index].pov_nccf;
    this_info = prev_info;
    if (this_info != NULL) ++iter;
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
  PitchFrameInfo *this_info = this;  // it will change in the loop.


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


// struct NccfInfo is used to cache certain quantities that we need for online
// operation, for the first "recompute_frame" frames of the file (e.g. 300);
// after that many frames, or after the user calls InputFinished(), we redo the
// initial backtraces, as we'll then have a better estimate of the average signal
// energy.
struct NccfInfo {

  Vector<BaseFloat> nccf_pitch_resampled;  // resampled nccf_pitch
  BaseFloat avg_norm_prod; // average value of e1 * e2.
  BaseFloat mean_square_energy;  // mean_square energy we used when computing the
                                 // original ballast term for
                                 // "nccf_pitch_resampled".

  NccfInfo(BaseFloat avg_norm_prod,
           BaseFloat mean_square_energy):
      avg_norm_prod(avg_norm_prod),
      mean_square_energy(mean_square_energy) { }
};



// We could inherit from OnlineBaseFeature as we have the same interface,
// but this will unnecessary force a lot of our functions to be virtual.
class OnlinePitchFeatureImpl {
 public:
  explicit OnlinePitchFeatureImpl(const PitchExtractionOptions &opts);

  int32 Dim() const { return 2; }

  BaseFloat FrameShiftInSeconds() const;

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
  /// Note: the number this returns depends on whether input_finished_ == true;
  /// if it is, it will "force out" a final frame or two.
  int32 NumFramesAvailable(int64 num_downsampled_samples, bool snip_edges) const;

  /// This function extracts from the signal the samples numbered from
  /// "sample_index" (numbered in the full downsampled signal, not just this
  /// part), and of length equal to window->Dim().  It uses the data members
  /// downsampled_samples_discarded_ and downsampled_signal_remainder_, as well
  /// as the more recent part of the downsampled wave "downsampled_wave_part"
  /// which is provided.
  ///
  /// @param downsampled_wave_part  One chunk of the downsampled wave,
  ///                      starting from sample-index downsampled_samples_discarded_.
  /// @param sample_index  The desired starting sample index (measured from
  ///                      the start of the whole signal, not just this part).
  /// @param window  The part of the signal is output to here.
  void ExtractFrame(const VectorBase<BaseFloat> &downsampled_wave_part,
                    int64 frame_index,
                    VectorBase<BaseFloat> *window);


  /// This function is called after we reach frame "recompute_frame", or when
  /// InputFinished() is called, whichever comes sooner.  It recomputes the
  /// backtraces for frames zero through recompute_frame, if needed because the
  /// average energy of the signal has changed, affecting the nccf ballast term.
  /// It works out the average signal energy from
  /// downsampled_samples_processed_, signal_sum_ and signal_sumsq_ (which, if
  /// you see the calling code, might include more frames than just
  /// "recompute_frame", it might include up to the end of the current chunk).
  void RecomputeBacktraces();


  /// This function updates downsampled_signal_remainder_,
  /// downsampled_samples_processed_, signal_sum_ and signal_sumsq_; it's called
  /// from AcceptWaveform().
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


  // nccf_info_ is indexed by frame-index, from frame 0 to at most
  // opts_.recompute_frame - 1.  It contains some information we'll
  // need to recompute the tracebacks after getting a better estimate
  // of the average energy of the signal.
  std::vector<NccfInfo*> nccf_info_;

  // Current number of frames which we can't output because Viterbi has not
  // converged for them, or opts_.max_frames_latency if we have reached that
  // limit.
  int32 frames_latency_;

  // The forward-cost at the current frame (the last frame in frame_info_);
  // this has the same dimension as lags_.  We normalize each time so
  // the lowest cost is zero, for numerical accuracy and so we can use float.
  Vector<BaseFloat> forward_cost_;

  // stores the constant part of forward_cost_.
  double forward_cost_remainder_;

  // The resampled-lag index and the NCCF (as computed for POV, without ballast
  // term) for each frame, as determined by Viterbi traceback from the best
  // final state.
  std::vector<std::pair<int32, BaseFloat> > lag_nccf_;

  bool input_finished_;

  /// sum-squared of previously processed parts of signal; used to get NCCF
  /// ballast term.  Denominator is downsampled_samples_processed_.
  double signal_sumsq_;

  /// sum of previously processed parts of signal; used to do mean-subtraction
  /// when getting sum-squared, along with signal_sumsq_.
  double signal_sum_;

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
    opts_(opts), forward_cost_remainder_(0.0), input_finished_(false),
    signal_sumsq_(0.0), signal_sum_(0.0), downsampled_samples_processed_(0) {
  signal_resampler_ = new LinearResample(opts.samp_freq, opts.resample_freq,
                                         opts.lowpass_cutoff,
                                         opts.lowpass_filter_width);

  double outer_min_lag = 1.0 / opts.max_f0 -
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / opts.min_f0 +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  nccf_first_lag_ = ceil(opts.resample_freq * outer_min_lag);
  nccf_last_lag_ = floor(opts.resample_freq * outer_max_lag);

  frames_latency_ = 0;  // will be set in AcceptWaveform()

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
    int64 num_downsampled_samples, bool snip_edges) const {
  int32 frame_shift = opts_.NccfWindowShift(),
      frame_length = opts_.NccfWindowSize();
  // Use the "full frame length" to compute the number
  // of frames only if the input is not finished.
  if (!input_finished_)
    frame_length += nccf_last_lag_;
  if (num_downsampled_samples < frame_length) {
    return 0;
  } else {
    if (!snip_edges) {
      if (input_finished_) {
        return static_cast<int32>(num_downsampled_samples * 1.0f /
                                  frame_shift + 0.5f);
      } else {
        return static_cast<int32>((num_downsampled_samples - frame_length / 2) *
                                   1.0f / frame_shift + 0.5f);
      }
    } else {
      return static_cast<int32>((num_downsampled_samples - frame_length) /
                                 frame_shift + 1);
    }
  }
}

void OnlinePitchFeatureImpl::UpdateRemainder(
    const VectorBase<BaseFloat> &downsampled_wave_part) {
  // frame_info_ has an extra element at frame-1, so subtract
  // one from the length.
  int64 num_frames = static_cast<int64>(frame_info_.size()) - 1,
      next_frame = num_frames,
      frame_shift = opts_.NccfWindowShift(),
      next_frame_sample = frame_shift * next_frame;

  signal_sumsq_ += VecVec(downsampled_wave_part, downsampled_wave_part);
  signal_sum_ += downsampled_wave_part.Sum();

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
      } else {  // in old remainder; only reach here if waveform supplied is
        new_remainder(i - next_frame_sample) =                      //  tiny.
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

  // Treat edge cases first
  if (sample_index < 0) {
    // Part of the frame is before the beginning of the signal. This
    // should only happen if opts_.snip_edges == false, when we are
    // processing the first few frames of signal. In this case
    // we pad with zeros.
    KALDI_ASSERT(opts_.snip_edges == false);
    int32 sub_frame_length = sample_index + full_frame_length;
    int32 sub_frame_index = full_frame_length - sub_frame_length;
    KALDI_ASSERT(sub_frame_length > 0 && sub_frame_index > 0);
    window->SetZero();
    SubVector<BaseFloat> sub_window(*window, sub_frame_index, sub_frame_length);
    ExtractFrame(downsampled_wave_part, 0, &sub_window);
    return;
  }

  if (offset + full_frame_length > downsampled_wave_part.Dim()) {
    // Requested frame is past end of the signal.  This should only happen if
    // input_finished_ == true, when we're flushing out the last couple of
    // frames of signal.  In this case we pad with zeros.
    KALDI_ASSERT(input_finished_);
    int32 sub_frame_length = downsampled_wave_part.Dim() - offset;
    KALDI_ASSERT(sub_frame_length > 0);
    window->SetZero();
    SubVector<BaseFloat> sub_window(*window, 0, sub_frame_length);
    ExtractFrame(downsampled_wave_part, sample_index, &sub_window);
    return;
  }

  // "offset" is the offset of the start of the frame, into this
  // signal.
  if (offset >= 0) {
    // frame is full inside the new part of the signal.
    window->CopyFromVec(downsampled_wave_part.Range(offset, full_frame_length));
  } else {
    // frame is partly in the remainder and partly in the new part.
    int32 remainder_offset = downsampled_signal_remainder_.Dim() + offset;
    KALDI_ASSERT(remainder_offset >= 0);  // or we didn't keep enough remainder.
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

BaseFloat OnlinePitchFeatureImpl::FrameShiftInSeconds() const {
  return opts_.frame_shift_ms / 1000.0f;
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
  // Process an empty waveform; this has an effect because
  // after setting input_finished_ to true, NumFramesAvailable()
  // will return a slightly larger number.
  AcceptWaveform(opts_.samp_freq, Vector<BaseFloat>());
  int32 num_frames = static_cast<size_t>(frame_info_.size() - 1);
  if (num_frames < opts_.recompute_frame && !opts_.nccf_ballast_online)
    RecomputeBacktraces();
  frames_latency_ = 0;
  KALDI_VLOG(3) << "Pitch-tracking Viterbi cost is "
                << (forward_cost_remainder_ / num_frames)
                << " per frame, over " << num_frames << " frames.";
}

// see comment with declaration.  This is only relevant for online
// operation (it gets called for non-online mode, but is a no-op).
void OnlinePitchFeatureImpl::RecomputeBacktraces() {
  KALDI_ASSERT(!opts_.nccf_ballast_online);
  int32 num_frames = static_cast<int32>(frame_info_.size()) - 1;

  // The assertion reflects how we believe this function will be called.
  KALDI_ASSERT(num_frames <= opts_.recompute_frame);
  KALDI_ASSERT(nccf_info_.size() == static_cast<size_t>(num_frames));
  if (num_frames == 0)
    return;
  double num_samp = downsampled_samples_processed_, sum = signal_sum_,
      sumsq = signal_sumsq_, mean = sum / num_samp;
  BaseFloat mean_square = sumsq / num_samp - mean * mean;

  bool must_recompute = false;
  BaseFloat threshold = 0.01;
  for (int32 frame = 0; frame < num_frames; frame++)
    if (!ApproxEqual(nccf_info_[frame]->mean_square_energy,
                     mean_square, threshold))
      must_recompute = true;

  if (!must_recompute) {
    // Nothing to do.  We'll reach here, for instance, if everything was in one
    // chunk and opts_.nccf_ballast_online == false.  This is the case for
    // offline processing.
    for (size_t i = 0; i < nccf_info_.size(); i++)
      delete nccf_info_[i];
    nccf_info_.clear();
    return;
  }

  int32 num_states = forward_cost_.Dim(),
      basic_frame_length = opts_.NccfWindowSize();

  BaseFloat new_nccf_ballast = pow(mean_square * basic_frame_length, 2) *
      opts_.nccf_ballast;

  double forward_cost_remainder = 0.0;
  Vector<BaseFloat> forward_cost(num_states),  // start off at zero.
      next_forward_cost(forward_cost);
  std::vector<std::pair<int32, int32 > > index_info;

  for (int32 frame = 0; frame < num_frames; frame++) {
    NccfInfo &nccf_info = *nccf_info_[frame];
    BaseFloat old_mean_square = nccf_info_[frame]->mean_square_energy,
        avg_norm_prod = nccf_info_[frame]->avg_norm_prod,
        old_nccf_ballast = pow(old_mean_square * basic_frame_length, 2) *
            opts_.nccf_ballast,
        nccf_scale = pow((old_nccf_ballast + avg_norm_prod) /
                         (new_nccf_ballast + avg_norm_prod),
                         static_cast<BaseFloat>(0.5));
    // The "nccf_scale" is an estimate of the scaling factor by which the NCCF
    // would change on this frame, on average, by changing the ballast term from
    // "old_nccf_ballast" to "new_nccf_ballast".  It's not exact because the
    // "avg_norm_prod" is just an average of the product e1 * e2 of frame
    // energies of the (frame, shifted-frame), but these won't change that much
    // within a frame, and even if they do, the inaccuracy of the scaled NCCF
    // will still be very small if the ballast term didn't change much, or if
    // it's much larger or smaller than e1*e2.  By doing it as a simple scaling,
    // we save the overhead of the NCCF resampling, which is a considerable part
    // of the whole computation.
    nccf_info.nccf_pitch_resampled.Scale(nccf_scale);

    frame_info_[frame + 1]->ComputeBacktraces(
        opts_, nccf_info.nccf_pitch_resampled, lags_,
        forward_cost, &index_info, &next_forward_cost);

    forward_cost.Swap(&next_forward_cost);
    BaseFloat remainder = forward_cost.Min();
    forward_cost_remainder += remainder;
    forward_cost.Add(-remainder);
  }
  KALDI_VLOG(3) << "Forward-cost per frame changed from "
                << (forward_cost_remainder_ / num_frames) << " to "
                << (forward_cost_remainder / num_frames);

  forward_cost_remainder_ = forward_cost_remainder;
  forward_cost_.Swap(&forward_cost);

  int32 best_final_state;
  forward_cost_.Min(&best_final_state);

  if (lag_nccf_.size() != static_cast<size_t>(num_frames))
    lag_nccf_.resize(num_frames);

  frame_info_.back()->SetBestState(best_final_state, lag_nccf_);
  frames_latency_ =
      frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
  for (size_t i = 0; i < nccf_info_.size(); i++)
    delete nccf_info_[i];
  nccf_info_.clear();
}

OnlinePitchFeatureImpl::~OnlinePitchFeatureImpl() {
  delete nccf_resampler_;
  delete signal_resampler_;
  for (size_t i = 0; i < frame_info_.size(); i++)
    delete frame_info_[i];
  for (size_t i = 0; i < nccf_info_.size(); i++)
    delete nccf_info_[i];
}

void OnlinePitchFeatureImpl::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &wave) {
  // flush out the last few samples of input waveform only if input_finished_ ==
  // true.
  const bool flush = input_finished_;

  Vector<BaseFloat> downsampled_wave;
  signal_resampler_->Resample(wave, flush, &downsampled_wave);

  // these variables will be used to compute the root-mean-square value of the
  // signal for the ballast term.
  double cur_sumsq = signal_sumsq_, cur_sum = signal_sum_;
  int64 cur_num_samp = downsampled_samples_processed_,
      prev_frame_end_sample = 0;
  if (!opts_.nccf_ballast_online) {
    cur_sumsq += VecVec(downsampled_wave, downsampled_wave);
    cur_sum += downsampled_wave.Sum();
    cur_num_samp += downsampled_wave.Dim();
  }

  // end_frame is the total number of frames we can now process, including
  // previously processed ones.
  int32 end_frame = NumFramesAvailable(
      downsampled_samples_processed_ + downsampled_wave.Dim(), opts_.snip_edges);
  // "start_frame" is the first frame-index we process
  int32 start_frame = frame_info_.size() - 1,
      num_new_frames = end_frame - start_frame;

  if (num_new_frames == 0) {
    UpdateRemainder(downsampled_wave);
    return;
    // continuing to the rest of the code would generate
    // an error when sizing matrices with zero rows, and
    // anyway is a waste of time.
  }

  int32 num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_,
      num_resampled_lags = lags_.Dim(),
      frame_shift = opts_.NccfWindowShift(),
      basic_frame_length = opts_.NccfWindowSize(),
      full_frame_length = basic_frame_length + nccf_last_lag_;

  Vector<BaseFloat> window(full_frame_length),
      inner_prod(num_measured_lags),
      norm_prod(num_measured_lags);
  Matrix<BaseFloat> nccf_pitch(num_new_frames, num_measured_lags),
      nccf_pov(num_new_frames, num_measured_lags);

  Vector<BaseFloat> cur_forward_cost(num_resampled_lags);


  // Because the resampling of the NCCF is more efficient when grouped together,
  // we first compute the NCCF for all frames, then resample as a matrix, then
  // do the Viterbi [that happens inside the constructor of PitchFrameInfo].

  for (int32 frame = start_frame; frame < end_frame; frame++) {
    // start_sample is index into the whole wave, not just this part.
    int64 start_sample;
    if (opts_.snip_edges) {
      // Usual case: offset starts at 0
      start_sample = static_cast<int64>(frame) * frame_shift;
    } else {
      // When we are not snipping the edges, the first offsets may be
      // negative. In this case we will pad with zeros, it should not impact
      // the pitch tracker.
      start_sample =
        static_cast<int64>((frame + 0.5) * frame_shift) - full_frame_length / 2;
    }
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
      if (end_sample > downsampled_wave.Dim()) {
        KALDI_ASSERT(input_finished_);
        end_sample = downsampled_wave.Dim();
      }
      SubVector<BaseFloat> new_part(downsampled_wave, prev_frame_end_sample,
                                    end_sample - prev_frame_end_sample);
      cur_num_samp += new_part.Dim();
      cur_sumsq += VecVec(new_part, new_part);
      cur_sum += new_part.Sum();
      prev_frame_end_sample = end_sample;
    }
    double mean_square = cur_sumsq / cur_num_samp -
        pow(cur_sum / cur_num_samp, 2.0);

    ComputeCorrelation(window, nccf_first_lag_, nccf_last_lag_,
                       basic_frame_length, &inner_prod, &norm_prod);
    double nccf_ballast_pov = 0.0,
        nccf_ballast_pitch = pow(mean_square * basic_frame_length, 2) *
             opts_.nccf_ballast,
        avg_norm_prod = norm_prod.Sum() / norm_prod.Dim();
    SubVector<BaseFloat> nccf_pitch_row(nccf_pitch, frame - start_frame);
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pitch,
                &nccf_pitch_row);
    SubVector<BaseFloat> nccf_pov_row(nccf_pov, frame - start_frame);
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pov,
                &nccf_pov_row);
    if (frame < opts_.recompute_frame)
      nccf_info_.push_back(new NccfInfo(avg_norm_prod, mean_square));
  }

  Matrix<BaseFloat> nccf_pitch_resampled(num_new_frames, num_resampled_lags);
  nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
  nccf_pitch.Resize(0, 0);  // no longer needed.
  Matrix<BaseFloat> nccf_pov_resampled(num_new_frames, num_resampled_lags);
  nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);
  nccf_pov.Resize(0, 0);  // no longer needed.

  // We've finished dealing with the waveform so we can call UpdateRemainder
  // now; we need to call it before we possibly call RecomputeBacktraces()
  // below, which is why we don't do it at the very end.
  UpdateRemainder(downsampled_wave);

  std::vector<std::pair<int32, int32 > > index_info;

  for (int32 frame = start_frame; frame < end_frame; frame++) {
    int32 frame_idx = frame - start_frame;
    PitchFrameInfo *prev_info = frame_info_.back(),
        *cur_info = new PitchFrameInfo(prev_info);
    cur_info->SetNccfPov(nccf_pov_resampled.Row(frame_idx));
    cur_info->ComputeBacktraces(opts_, nccf_pitch_resampled.Row(frame_idx),
                                lags_, forward_cost_, &index_info,
                                &cur_forward_cost);
    forward_cost_.Swap(&cur_forward_cost);
    // Renormalize forward_cost so smallest element is zero.
    BaseFloat remainder = forward_cost_.Min();
    forward_cost_remainder_ += remainder;
    forward_cost_.Add(-remainder);
    frame_info_.push_back(cur_info);
    if (frame < opts_.recompute_frame)
      nccf_info_[frame]->nccf_pitch_resampled =
          nccf_pitch_resampled.Row(frame_idx);
    if (frame == opts_.recompute_frame - 1 && !opts_.nccf_ballast_online)
      RecomputeBacktraces();
  }

  // Trace back the best-path.
  int32 best_final_state;
  forward_cost_.Min(&best_final_state);
  lag_nccf_.resize(frame_info_.size() - 1);  // will keep any existing data.
  frame_info_.back()->SetBestState(best_final_state, lag_nccf_);
  frames_latency_ =
      frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
  KALDI_VLOG(4) << "Latency is " << frames_latency_;
}



// Some functions that forward from OnlinePitchFeature to
// OnlinePitchFeatureImpl.
int32 OnlinePitchFeature::NumFramesReady() const {
  return impl_->NumFramesReady();
}

OnlinePitchFeature::OnlinePitchFeature(const PitchExtractionOptions &opts)
    :impl_(new OnlinePitchFeatureImpl(opts)) { }

bool OnlinePitchFeature::IsLastFrame(int32 frame) const {
  return impl_->IsLastFrame(frame);
}

BaseFloat OnlinePitchFeature::FrameShiftInSeconds() const {
  return impl_->FrameShiftInSeconds();
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


/**
   This function is called from ComputeKaldiPitch when the user
   specifies opts.simulate_first_pass_online == true.  It gives
   the "first-pass" version of the features, which you would get
   on the first decoding pass in an online setting.  These may
   differ slightly from the final features due to both the
   way the Viterbi traceback works (this is affected by
   opts.max_frames_latency), and the online way we compute
   the average signal energy.
*/
void ComputeKaldiPitchFirstPass(
    const PitchExtractionOptions &opts,
    const VectorBase<BaseFloat> &wave,
    Matrix<BaseFloat> *output) {

  int32 cur_rows = 100;
  Matrix<BaseFloat> feats(cur_rows, 2);

  OnlinePitchFeature pitch_extractor(opts);
  KALDI_ASSERT(opts.frames_per_chunk > 0 &&
               "--simulate-first-pass-online option does not make sense "
               "unless you specify --frames-per-chunk");

  int32 cur_offset = 0, cur_frame = 0, samp_per_chunk =
      opts.frames_per_chunk * opts.samp_freq * opts.frame_shift_ms / 1000.0f;

  while (cur_offset < wave.Dim()) {
    int32 num_samp = std::min(samp_per_chunk, wave.Dim() - cur_offset);
    SubVector<BaseFloat> wave_chunk(wave, cur_offset, num_samp);
    pitch_extractor.AcceptWaveform(opts.samp_freq, wave_chunk);
    cur_offset += num_samp;
    if (cur_offset == wave.Dim())
      pitch_extractor.InputFinished();
    // Get each frame as soon as it is ready.
    for (; cur_frame < pitch_extractor.NumFramesReady(); cur_frame++) {
      if (cur_frame >= cur_rows) {
        cur_rows *= 2;
        feats.Resize(cur_rows, 2, kCopyData);
      }
      SubVector<BaseFloat> row(feats, cur_frame);
      pitch_extractor.GetFrame(cur_frame, &row);
    }
  }
  if (cur_frame  == 0) {
    KALDI_WARN << "No features output since wave file too short";
    output->Resize(0, 0);
  } else {
    *output = feats.RowRange(0, cur_frame);
  }
}



void ComputeKaldiPitch(const PitchExtractionOptions &opts,
                       const VectorBase<BaseFloat> &wave,
                       Matrix<BaseFloat> *output) {
  if (opts.simulate_first_pass_online) {
    ComputeKaldiPitchFirstPass(opts, wave, output);
    return;
  }
  OnlinePitchFeature pitch_extractor(opts);

  if (opts.frames_per_chunk == 0) {
    pitch_extractor.AcceptWaveform(opts.samp_freq, wave);
  } else {
    // the user may set opts.frames_per_chunk for better compatibility with
    // online operation.
    KALDI_ASSERT(opts.frames_per_chunk > 0);
    int32 cur_offset = 0, samp_per_chunk =
        opts.frames_per_chunk * opts.samp_freq * opts.frame_shift_ms / 1000.0f;
    while (cur_offset < wave.Dim()) {
      int32 num_samp = std::min(samp_per_chunk, wave.Dim() - cur_offset);
      SubVector<BaseFloat> wave_chunk(wave, cur_offset, num_samp);
      pitch_extractor.AcceptWaveform(opts.samp_freq, wave_chunk);
      cur_offset += num_samp;
    }
  }
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


/*
  This comment describes our invesigation of how much latency the
  online-processing algorithm introduces, i.e. how many frames you would
  typically have to wait until the traceback converges, if you were to set the
  --max-frames-latency to a very large value.

  This was done on a couple of files of language-id data.

  /home/dpovey/kaldi-online/src/featbin/compute-kaldi-pitch-feats --frames-per-chunk=10 --max-frames-latency=100 --verbose=4 --sample-frequency=8000 --resample-frequency=2600 "scp:head -n 2 data/train/wav.scp |" ark:/dev/null 2>&1   | grep Latency | wc
   4871   24355  443991
 /home/dpovey/kaldi-online/src/featbin/compute-kaldi-pitch-feats --frames-per-chunk=10 --max-frames-latency=100 --verbose=4 --sample-frequency=8000 --resample-frequency=2600 "scp:head -n 2 data/train/wav.scp |" ark:/dev/null 2>&1  | grep Latency | grep 100 | wc
   1534    7670  141128

# as above, but with 50 instead of 10 in the --max-frames-latency and grep statements.
   2070   10350  188370
# as above, but with 10 instead of 50.
   4067   20335  370097

   This says that out of 4871 selected frames [we measured the latency every 10
   frames, since --frames-per-chunk=10], in 1534 frames (31%), the latency was
    >= 100 frames, i.e. >= 1 second.  Including the other numbers, we can see
    that

    31% of frames had latency >= 1 second
    42% of frames had latency >= 0.5 second
    83% of frames had latency >= 0.1 second.

  This doesn't necessarily mean that we actually have a latency of >= 1 second 31% of
  the time when using these features, since by using the --max-frames-latency option
  (default: 30 frames), it will limit the latency to, say, 0.3 seconds, and trace back
  from the best current pitch.  Most of the time this will probably cause no change in
  the pitch traceback since the best current pitch is probably the "right" point to
  trace back from.  And anyway, in the online-decoding, we will most likely rescore
  the features at the end anyway, and the traceback gets recomputed, so there will
  be no inaccuracy (assuming the first-pass lattice had everything we needed).

  Probably the greater source of inaccuracy due to the online algorithm is the
  online energy-normalization, which affects the NCCF-ballast term, and which,
  for reasons of efficiency, we don't attempt to "correct" in a later rescoring
  pass.  This will make the most difference in the first few frames of the file,
  before the first voicing, where it will tend to produce more pitch movement
  than the offline version of the algorithm.
*/


// Function to do data accumulation for on-line usage
template<typename Real>
inline void AppendVector(const VectorBase<Real> &src, Vector<Real> *dst) {
  if (src.Dim() == 0) return;
  dst->Resize(dst->Dim() + src.Dim(), kCopyData);
  dst->Range(dst->Dim() - src.Dim(), src.Dim()).CopyFromVec(src);
}

/**
   Note on the implementation of OnlineProcessPitch: the
   OnlineFeatureInterface allows random access to features (i.e. not necessarily
   sequential order), so we need to support that.  But we don't need to support
   it very efficiently, and our implementation is most efficient if frames are
   accessed in sequential order.

   Also note: we have to be a bit careful in this implementation because
   the input features may change.  That is: if we call
   src_->GetFrame(t, &vec) from GetFrame(), we can't guarantee that a later
   call to src_->GetFrame(t, &vec) from another GetFrame() will return the
   same value.  In fact, while designing this class we used some knowledge
   of how the OnlinePitchFeature class works to minimize the amount of
   re-querying we had to do.
*/
OnlineProcessPitch::OnlineProcessPitch(
    const ProcessPitchOptions &opts,
    OnlineFeatureInterface *src):
    opts_(opts), src_(src),
    dim_ ((opts.add_pov_feature ? 1 : 0)
          + (opts.add_normalized_log_pitch ? 1 : 0)
          + (opts.add_delta_pitch ? 1 : 0)
          + (opts.add_raw_log_pitch ? 1 : 0)) {
  KALDI_ASSERT(dim_ > 0 &&
               " At least one of the pitch features should be chosen. "
               "Check your post-process-pitch options.");
  KALDI_ASSERT(src->Dim() == kRawFeatureDim &&
               "Input feature must be pitch feature (should have dimension 2)");
}


void OnlineProcessPitch::GetFrame(int32 frame,
                                  VectorBase<BaseFloat> *feat) {
  int32 frame_delayed = frame < opts_.delay ? 0 : frame - opts_.delay;
  KALDI_ASSERT(feat->Dim() == dim_ &&
               frame_delayed < NumFramesReady());
  int32 index = 0;
  if (opts_.add_pov_feature)
    (*feat)(index++) = GetPovFeature(frame_delayed);
  if (opts_.add_normalized_log_pitch)
    (*feat)(index++) = GetNormalizedLogPitchFeature(frame_delayed);
  if (opts_.add_delta_pitch)
    (*feat)(index++) = GetDeltaPitchFeature(frame_delayed);
  if (opts_.add_raw_log_pitch)
    (*feat)(index++) = GetRawLogPitchFeature(frame_delayed);
  KALDI_ASSERT(index == dim_);
}

BaseFloat OnlineProcessPitch::GetPovFeature(int32 frame) const {
  Vector<BaseFloat> tmp(kRawFeatureDim);
  src_->GetFrame(frame, &tmp);  // (NCCF, pitch) from pitch extractor
  BaseFloat nccf = tmp(0);
  return opts_.pov_scale * NccfToPovFeature(nccf)
      + opts_.pov_offset;
}

BaseFloat OnlineProcessPitch::GetDeltaPitchFeature(int32 frame) {
  // Rather than computing the delta pitch directly in code here,
  // which might seem easier, we accumulate a small window of features
  // and call ComputeDeltas.  This might seem like overkill; the reason
  // we do it this way is to ensure that the end effects (at file
  // beginning and end) are handled in a consistent way.
  int32 context = opts_.delta_window;
  int32 start_frame = std::max(0, frame - context),
      end_frame = std::min(frame + context + 1, src_->NumFramesReady()),
      frames_in_window = end_frame - start_frame;
  Matrix<BaseFloat> feats(frames_in_window, 1),
      delta_feats;

  for (int32 f = start_frame; f < end_frame; f++)
    feats(f - start_frame, 0) = GetRawLogPitchFeature(f);

  DeltaFeaturesOptions delta_opts;
  delta_opts.order = 1;
  delta_opts.window = opts_.delta_window;
  ComputeDeltas(delta_opts, feats, &delta_feats);
  while (delta_feature_noise_.size() <= static_cast<size_t>(frame)) {
    delta_feature_noise_.push_back(RandGauss() *
                                   opts_.delta_pitch_noise_stddev);
  }
  // note: delta_feats will have two columns, second contains deltas.
  return (delta_feats(frame - start_frame, 1) + delta_feature_noise_[frame]) *
      opts_.delta_pitch_scale;
}

BaseFloat OnlineProcessPitch::GetRawLogPitchFeature(int32 frame) const {
  Vector<BaseFloat> tmp(kRawFeatureDim);
  src_->GetFrame(frame, &tmp);
  BaseFloat pitch = tmp(1);
  KALDI_ASSERT(pitch > 0);
  return Log(pitch);
}

BaseFloat OnlineProcessPitch::GetNormalizedLogPitchFeature(int32 frame) {
  UpdateNormalizationStats(frame);
  BaseFloat log_pitch = GetRawLogPitchFeature(frame),
      avg_log_pitch = normalization_stats_[frame].sum_log_pitch_pov /
        normalization_stats_[frame].sum_pov,
      normalized_log_pitch = log_pitch - avg_log_pitch;
  return normalized_log_pitch * opts_.pitch_scale;
}


// inline
void OnlineProcessPitch::GetNormalizationWindow(int32 t,
                                                int32 src_frames_ready,
                                                int32 *window_begin,
                                                int32 *window_end) const {
  int32 left_context = opts_.normalization_left_context;
  int32 right_context = opts_.normalization_right_context;
  *window_begin = std::max(0, t - left_context);
  *window_end = std::min(t + right_context + 1, src_frames_ready);
}


// Makes sure the entry in normalization_stats_ for this frame is up to date;
// called from GetNormalizedLogPitchFeature.
// the cur_num_frames and input_finished variables are needed because the
// pitch features for a given frame may change as we see more data.
void OnlineProcessPitch::UpdateNormalizationStats(int32 frame) {
  KALDI_ASSERT(frame >= 0);
  if (normalization_stats_.size() <= frame)
    normalization_stats_.resize(frame + 1);
  int32 cur_num_frames = src_->NumFramesReady();
  bool input_finished = src_->IsLastFrame(cur_num_frames - 1);

  NormalizationStats &this_stats = normalization_stats_[frame];
  if (this_stats.cur_num_frames == cur_num_frames &&
      this_stats.input_finished == input_finished) {
    // Stats are fully up-to-date.
    return;
  }
  int32 this_window_begin, this_window_end;
  GetNormalizationWindow(frame, cur_num_frames,
                         &this_window_begin, &this_window_end);

  if (frame > 0) {
    const NormalizationStats &prev_stats = normalization_stats_[frame - 1];
    if (prev_stats.cur_num_frames == cur_num_frames &&
        prev_stats.input_finished == input_finished) {
      // we'll derive this_stats efficiently from prev_stats.
      // Checking that cur_num_frames and input_finished have not changed
      // ensures that the underlying features will not have changed.
      this_stats = prev_stats;
      int32 prev_window_begin, prev_window_end;
      GetNormalizationWindow(frame - 1, cur_num_frames,
                             &prev_window_begin, &prev_window_end);
      if (this_window_begin != prev_window_begin) {
        KALDI_ASSERT(this_window_begin == prev_window_begin + 1);
        Vector<BaseFloat> tmp(kRawFeatureDim);
        src_->GetFrame(prev_window_begin, &tmp);
        BaseFloat accurate_pov = NccfToPov(tmp(0)),
            log_pitch = Log(tmp(1));
        this_stats.sum_pov -= accurate_pov;
        this_stats.sum_log_pitch_pov -= accurate_pov * log_pitch;
      }
      if (this_window_end != prev_window_end) {
        KALDI_ASSERT(this_window_end == prev_window_end + 1);
        Vector<BaseFloat> tmp(kRawFeatureDim);
        src_->GetFrame(prev_window_end, &tmp);
        BaseFloat accurate_pov = NccfToPov(tmp(0)),
            log_pitch = Log(tmp(1));
        this_stats.sum_pov += accurate_pov;
        this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
      }
      return;
    }
  }
  // The way we do it here is not the most efficient way to do it;
  // we'll see if it becomes a problem.  The issue is we have to redo
  // this computation from scratch each time we process a new chunk, which
  // may be a little inefficient if the chunk-size is very small.
  this_stats.cur_num_frames = cur_num_frames;
  this_stats.input_finished = input_finished;
  this_stats.sum_pov = 0.0;
  this_stats.sum_log_pitch_pov = 0.0;
  Vector<BaseFloat> tmp(kRawFeatureDim);
  for (int32 f = this_window_begin; f < this_window_end; f++) {
    src_->GetFrame(f, &tmp);
    BaseFloat accurate_pov = NccfToPov(tmp(0)),
        log_pitch = Log(tmp(1));
    this_stats.sum_pov += accurate_pov;
    this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
  }
}

int32 OnlineProcessPitch::NumFramesReady() const {
  int32 src_frames_ready = src_->NumFramesReady();
  if (src_frames_ready == 0) {
    return 0;
  } else if (src_->IsLastFrame(src_frames_ready - 1)) {
    return src_frames_ready + opts_.delay;
  } else {
    return std::max(0, src_frames_ready -
      opts_.normalization_right_context + opts_.delay);
  }
}

void ProcessPitch(const ProcessPitchOptions &opts,
                  const MatrixBase<BaseFloat> &input,
                  Matrix<BaseFloat> *output) {
  OnlineMatrixFeature pitch_feat(input);

  OnlineProcessPitch online_process_pitch(opts, &pitch_feat);

  output->Resize(online_process_pitch.NumFramesReady(),
                 online_process_pitch.Dim());
  for (int32 t = 0; t < online_process_pitch.NumFramesReady(); t++) {
    SubVector<BaseFloat> row(*output, t);
    online_process_pitch.GetFrame(t, &row);
  }
}


void ComputeAndProcessKaldiPitch(
    const PitchExtractionOptions &pitch_opts,
    const ProcessPitchOptions &process_opts,
    const VectorBase<BaseFloat> &wave,
    Matrix<BaseFloat> *output) {

  OnlinePitchFeature pitch_extractor(pitch_opts);

  if (pitch_opts.simulate_first_pass_online) {
    KALDI_ASSERT(pitch_opts.frames_per_chunk > 0 &&
                 "--simulate-first-pass-online option does not make sense "
                 "unless you specify --frames-per-chunk");
  }

  OnlineProcessPitch post_process(process_opts, &pitch_extractor);

  int32 cur_rows = 100;
  Matrix<BaseFloat> feats(cur_rows, post_process.Dim());

  int32 cur_offset = 0, cur_frame = 0,
      samp_per_chunk = pitch_opts.frames_per_chunk *
      pitch_opts.samp_freq * pitch_opts.frame_shift_ms / 1000.0f;

  // We request the first-pass features as soon as they are available,
  // regardless of whether opts.simulate_first_pass_online == true.  If
  // opts.simulate_first_pass_online == true this should
  // not affect the features generated, but it helps us to test the code
  // in a way that's closer to what online decoding would see.

  while (cur_offset < wave.Dim()) {
    int32 num_samp;
    if (samp_per_chunk > 0)
      num_samp = std::min(samp_per_chunk, wave.Dim() - cur_offset);
    else  // user left opts.frames_per_chunk at zero.
      num_samp = wave.Dim();
    SubVector<BaseFloat> wave_chunk(wave, cur_offset, num_samp);
    pitch_extractor.AcceptWaveform(pitch_opts.samp_freq, wave_chunk);
    cur_offset += num_samp;
    if (cur_offset == wave.Dim())
      pitch_extractor.InputFinished();

    // Get each frame as soon as it is ready.
    for (; cur_frame < post_process.NumFramesReady(); cur_frame++) {
      if (cur_frame >= cur_rows) {
        cur_rows *= 2;
        feats.Resize(cur_rows, post_process.Dim(), kCopyData);
      }
      SubVector<BaseFloat> row(feats, cur_frame);
      post_process.GetFrame(cur_frame, &row);
    }
  }

  if (pitch_opts.simulate_first_pass_online) {
    if (cur_frame == 0) {
      KALDI_WARN << "No features output since wave file too short";
      output->Resize(0, 0);
    } else {
      *output = feats.RowRange(0, cur_frame);
    }
  } else {
    // want the "final" features for second pass, so get them again.
    output->Resize(post_process.NumFramesReady(), post_process.Dim());
    for (int32 frame = 0; frame < post_process.NumFramesReady(); frame++) {
      SubVector<BaseFloat> row(*output, frame);
      post_process.GetFrame(frame, &row);
    }
  }
}


}  // namespace kaldi

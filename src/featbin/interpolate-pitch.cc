// featbin/interpolate-pitch.cc

// Copyright 2013   Bagher BabaAli
//                  Johns Hopkins University (author: Daniel Povey)
//
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

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

struct PitchInterpolatorOptions {
  BaseFloat pitch_interval; // Discretization interval [affects efficiency]
  BaseFloat interpolator_factor; // This affects the tendency of the algorithm to
  // follow the observed pitch contours versus pick its own path which will tend
  // to be closer to a straight line.
  BaseFloat max_voicing_prob; // p(voicing) we use at the end of the range when it was observed
  // at one. (probably 0.9 is suitable; allows to not follow observed pitch even if p(voicing)=1.
  BaseFloat max_pitch_change_per_frame;
  PitchInterpolatorOptions(): pitch_interval(4.0),
                              interpolator_factor(1.0e-05),
                              max_voicing_prob(0.9),
                              max_pitch_change_per_frame(10.0) { }
  void Register(OptionsItf *opts) {
    opts->Register("pitch-interval", &pitch_interval, "Frequency interval in Hz, used "
                   "for the pitch interpolation and smoothing algorithm.");
    opts->Register("interpolator-factor", &interpolator_factor, "Factor affecting the "
                   "interpolation algorithm; setting it closer to zero will cause "
                   "it to follow the measured pitch more faithfully but less "
                   "smoothly");
    opts->Register("max-voicing-prob", &max_voicing_prob, "Probability of voicing the "
                   "algorithm uses as the observed p(voicing) approaches 1; having "
                   "value <1 allows it to interpolate even if p(voicing) = 1");
    opts->Register("max-pitch-change-per-frame", &max_pitch_change_per_frame, 
                   "This value should be set large enough to no longer affect the "
                   "results, but the larger it is the slower the algorithm will be.");
  }
  void Check() const {
    KALDI_ASSERT(pitch_interval > 0.0 && pitch_interval < 20.0 &&
                 interpolator_factor > 0.0 && interpolator_factor < 1.0 &&
                 max_voicing_prob <= 1.0 && max_voicing_prob >= 0.5 &&
                 max_pitch_change_per_frame > 2.0 * pitch_interval);
  }
};

struct PitchInterpolatorStats {
  int64 num_frames_tot;
  int64 num_frames_zero; // #frames that were zero in original pitch.
  int64 num_frames_changed; // #frames that were not zero originally, but
  // which the algorithm changed.
  
  PitchInterpolatorStats(): num_frames_tot(0), num_frames_zero(0),
                            num_frames_changed(0) { }
  void Print() {
    BaseFloat zero_percent = num_frames_zero * 100.0 / num_frames_tot,
        changed_percent = num_frames_changed * 100.0 / num_frames_tot;
        KALDI_LOG << "Over " << num_frames_tot << " frames, "
                  << zero_percent << "% were zero at input, and "
                  << changed_percent << "% were not zero but were changed.";
  }
};

class PitchInterpolator {
 public:
  PitchInterpolator(const PitchInterpolatorOptions &opts,
                    Matrix<BaseFloat> *mat,
                    PitchInterpolatorStats *stats):
      opts_(opts) {
    opts.Check();
    InitValues(*mat);
    Forward();
    Backtrace(mat, stats);
  }
 private:
  void InitValues(const Matrix<BaseFloat> &mat) {
    BaseFloat pitch_interval = opts_.pitch_interval;
    num_frames_ = mat.NumRows();
    KALDI_ASSERT(mat.NumCols() == 2);
    BaseFloat min_pitch = 1.0e+10, max_pitch = 0.0;
    pitch_.resize(num_frames_);
    p_voicing_.resize(num_frames_);
    for (int32 f = 0; f < num_frames_; f++) {
      BaseFloat p_voicing = mat(f, 0), pitch = mat(f, 1);
      p_voicing *= opts_.max_voicing_prob;
      if (pitch == 0.0) {
        p_voicing = 0.0; // complete uncertainty about real pitch.
      } else {
        if (pitch < min_pitch) min_pitch = pitch;
        if (pitch > max_pitch) max_pitch = pitch;
      }
      p_voicing_[f] = p_voicing;
    }
    if (max_pitch == 0.0) { // No voiced frames at all.
      min_pitch = 100.0;
      max_pitch = 100.0;
    }
    if (max_pitch <= min_pitch + (2.0 * pitch_interval)) {
      max_pitch = min_pitch + 2.0 * pitch_interval;
    } // avoid crashes.

    // Note: the + 2 here is for edge effects.
    num_pitches_ = floor((max_pitch - min_pitch) / pitch_interval + 0.5) + 2;
    KALDI_ASSERT(num_pitches_ >= 3);
    min_pitch_.resize(num_frames_);
    for (int32 f = 0; f < num_frames_; f++) {
      min_pitch_[f] = min_pitch - pitch_interval * RandUniform(); // bottom of
      // discretization range for each frame is randomly different.
      
      BaseFloat pitch = mat(f, 1);
      if (pitch == 0.0) {
        pitch_[f] = 0; // This will actually be a don't-care value; we just put in
        // some value that won't crash the algorithm.
      } else {
        int32 int_pitch = floor((pitch - min_pitch_[f]) / pitch_interval + 0.5);
        KALDI_ASSERT(int_pitch >= 0 && int_pitch < num_pitches_);
        pitch_[f] = int_pitch;
      }
    }
  }

  void MultiplyObsProb(int32 t) {
    // For the forward computation:
    // Multiplies the observation probabilities into alpha at time t.
    // constant_prob is the constant part that does not depend on the pitch value:
    BaseFloat constant_prob = (1.0 - p_voicing_[t]) * opts_.interpolator_factor,
        specified_prob = p_voicing_[t] + constant_prob;
    // specified_prob adds in the extra probability mass at the observed pitch value.
    BaseFloat log_constant_prob = Log(constant_prob),
        log_ratio = Log(specified_prob / constant_prob);
    log_alpha_.Add(log_constant_prob); // add log_constant_prob to all pitches at this time.
    
    log_alpha_(pitch_[t]) += log_ratio; // corrects this to be like adding
    // log(specified_prob) to the observed pitch at this time.  Note: if pitch_[t] == 0,
    // this won't have any effect because log_ratio will be zero too.
    
    Vector<BaseFloat> temp_rand(num_pitches_);
    temp_rand.SetRandn(); // Set to Gaussian noise.  Type of noise doesn't really matter.
    log_alpha_.AddVec(0.01, temp_rand); // We add a small amount of noise to the
    // observation probabilities; this has the effect of breaking symmetries in
    // a more random way to overcome certain weirdnesses that could otherwise
    // happen due to the discretization.
  }

  // This function updates log_alpha_, as a function of prev_log_alpha_; it also
  // updates back_pointers_[t];
  void ComputeTransitionProb(int32 t) {
    KALDI_ASSERT(t > 0);
    BaseFloat pitch_interval = opts_.pitch_interval;
    back_pointers_[t].resize(num_pitches_);
    
    // Transition probability between pitch p and p' on times t-1 and t
    // is (p - p')^2, with the pitch measured in Hz.  We're doing Viterbi,
    // so always pick the max over the previous frame's t.
    KALDI_ASSERT(t > 0 && t < num_frames_);
    int32 K = floor(opts_.max_pitch_change_per_frame / pitch_interval + 0.5);
    // K is max #bins we can move; a kind of pruning, for speed.
    for (int32 p = 0; p < num_pitches_; p++) {
      int32 min_prev_p = p - K, max_prev_p = p + K;
      if (min_prev_p < 0) min_prev_p = 0;
      if (max_prev_p >= num_pitches_) max_prev_p = num_pitches_ - 1;
      BaseFloat best_logprob = -1.0e+10;
      int32 best_prev_p = -1;
      for (int32 prev_p = min_prev_p; prev_p <= max_prev_p; prev_p++) {
        BaseFloat delta_pitch = (min_pitch_[t-1] + prev_p * pitch_interval) -
            (min_pitch_[t] + p * pitch_interval);
        BaseFloat this_logprob = prev_log_alpha_(prev_p) 
            - 0.5 * delta_pitch * delta_pitch;
        if (this_logprob > best_logprob) {
          best_logprob = this_logprob;
          best_prev_p = prev_p;
        }
      }
      back_pointers_[t][p] = best_prev_p;
      log_alpha_(p) = best_logprob;
    }    
  }
  
  void Forward() {
    // Viterbi in a discrete model of the pitch, in which the observation
    // probability of a pitch is p(voicing) at the observed pitch, and
    // interpolator_factor_ * 1.0 - p(voicing) at all other pitches.  the
    // transition log-probability is -0.5 times the squared difference in pitch.
    // [We measure this in Hz, not in integer values, to make it more invariant
    // to the discretization interval].

    back_pointers_.resize(num_frames_);

    log_alpha_.Resize(num_pitches_);
    prev_log_alpha_.Resize(num_pitches_);
    log_alpha_.Set(0.0);
    MultiplyObsProb(0);
    for (int32 t = 1; t < num_frames_; t++) {
      log_alpha_.Swap(&prev_log_alpha_);
      ComputeTransitionProb(t);
      MultiplyObsProb(t);
    }
  }
  void Backtrace(Matrix<BaseFloat> *mat, PitchInterpolatorStats *stats) {
    const BaseFloat pitch_interval = opts_.pitch_interval;
    BaseFloat *p_begin = log_alpha_.Data(), *p_end = p_begin + num_pitches_,
        *p_best = std::max_element(p_begin, p_end);

    std::vector<int32> best_pitch(num_frames_);
    int32 best_p = p_best - p_begin; // best discrete pitch p at time T-1.
    for (int32 t = num_frames_ - 1; t >= 0; t--) {
      { // Update stats:
        stats->num_frames_tot++;
        if (pitch_[t] == 0) stats->num_frames_zero++;
        else if (best_p != pitch_[t]) stats->num_frames_changed++;
      }
      BaseFloat pitch = min_pitch_[t] + pitch_interval * best_p;
      (*mat)(t, 1) = pitch;
      KALDI_ASSERT(best_p >= 0 && best_p < num_pitches_);
      if (t > 0)
        best_p = back_pointers_[t][best_p];
    }
  }
  const PitchInterpolatorOptions &opts_;
  std::vector<BaseFloat> min_pitch_; // Bottom of discretization range...
  // previously this was a BaseFloat, but for better pseudo-randomization we
  // have a slightly perturbed value for each frame now, so it's a vector.
  int32 num_frames_; // number of frames;
  int32 num_pitches_; // Number of discrete pitch intervals.
  std::vector<int32> pitch_; // observed pitch, discretized; [it's don't-care if algorithm had no
  // observation (0)]
  std::vector<BaseFloat> p_voicing_; // p(voicing) times max_voicing_prob_; or zero if
  // pitch was 0.0 for this frame.
  std::vector<std::vector<int32> > back_pointers_; // at each t, points to best pitch
  // on time t-1.

  Vector<BaseFloat> log_alpha_;
  Vector<BaseFloat> prev_log_alpha_;
};



// Linear Interpolation for places where the pitch value is zero 
void LinearlyInterpolatePitch(Matrix<BaseFloat> *mat) {
  int32 num_frames = mat->NumRows();
  int i = 0;
  Matrix<BaseFloat> &features = *mat;
  while (i < num_frames) {
    if(features(i, 1) == 0.0) {
      int start = i - 1;
      int end = i;
      while( (features(end, 1)) == 0.0 && (end < num_frames))
        end++;
      BaseFloat end_value = -1, start_value = -1;
      if (end < num_frames) end_value = features(end, 1);
      if (start > 0) start_value = features(start, 1);

      if (start_value < 0 && end_value < 0) {
        // the whole file is unvoiced -> just put an arbitrary value,
        // it will all be normalized out anyway.
        start_value = 1.0;
        end_value = 1.0;
      }
      // If we don't have a value for one end of the range, i.e. at the start or
      // end, set it to 0.9 times the pitch value that we have at the other end
      // of the range.  The reason we don't set it to that value itself, is that
      // then over this segment we would have zero time-derivative, so if we
      // took time derivatives we would have an artificial spike at zero.
      if (start_value < 0.0) start_value = 0.9 * end_value;
      if (end_value < 0.0) end_value = 0.9 * start_value;
      
      for(int k = start + 1; k < end; k++)
        features(k, 1) = start_value +
            (end_value - start_value) / (end - start) * (k - start);
      i = end;
    }
    i++;
  }
}


} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "This is a rather special-purpose program which processes 2-dimensional\n"
        "features consisting of (prob-of-voicing, pitch).  By default we do model-based\n"
        "pitch smoothing and interpolation (see code), or if --linear-interpolation=true,\n"
        "just linear interpolation across gaps where pitch == 0 (not predicted).\n"
        "Usage:  interpolate-pitch [options...] <feats-rspecifier> <feats-wspecifier>\n";

    
    // construct all the global objects
    ParseOptions opts(usage);

    bool linear_interpolation = false;
    PitchInterpolatorOptions interpolate_opts;
    
    opts.Register("linear-interpolation",
                &linear_interpolation, "If true, just do simple linear "
                "interpolation across gaps (else, model-based)");
    interpolate_opts.Register(&opts);
    
    // parse options (+filling the registered variables)
    opts.Read(argc, argv);

    if (opts.NumArgs() != 2) {
      opts.PrintUsage();
      exit(1);
    }
    
    std::string input_rspecifier = opts.GetArg(1);
    std::string output_wspecifier = opts.GetArg(2);

    SequentialBaseFloatMatrixReader reader(input_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.

    if (!kaldi_writer.Open(output_wspecifier))
       KALDI_ERR << "Could not initialize output with wspecifier "
                << output_wspecifier;

    int32 num_done = 0, num_err = 0;
    PitchInterpolatorStats stats;
    
    for (; !reader.Done(); reader.Next()) {
      std::string utt = reader.Key();   
      Matrix<BaseFloat> features = reader.Value();
      int num_frames = features.NumRows();

      if (num_frames == 0 && features.NumCols() != 2) {
        KALDI_WARN << "Feature file has bad size "
                   << features.NumRows() << " by " << features.NumCols();
        num_err++;
        continue;
      }
      
      if (linear_interpolation) LinearlyInterpolatePitch(&features);
      else {
        // work happens in constructor of this class.
        PitchInterpolator pi(interpolate_opts, &features, &stats);
      }
      kaldi_writer.Write(utt, features);
      num_done++;
        
      if (num_done % 10 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
    }
    if (!linear_interpolation) stats.Print();
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err
              << " with errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


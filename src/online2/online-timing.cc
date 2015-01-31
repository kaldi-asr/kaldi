// online2/online-timing.cc

// Copyright    2014  Johns Hopkins University (author: Daniel Povey)

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

#include "online2/online-timing.h"

namespace kaldi {

OnlineTimingStats::OnlineTimingStats():
    num_utts_(0), total_audio_(0.0), total_time_taken_(0.0),
    total_time_waited_(0.0), max_delay_(0.0) {
}

void OnlineTimingStats::Print(bool online){
  if (online) {
    double real_time_factor = total_time_taken_ / total_audio_,
        average_wait = (total_time_taken_ - total_audio_) / num_utts_,
        idle_proportion = total_time_waited_ / total_audio_,
        idle_percent = 100.0 * idle_proportion;

    KALDI_LOG << "Timing stats: real-time factor was " << real_time_factor
              << " (note: this cannot be less than one.)";
    KALDI_LOG << "Average delay was " << average_wait << " seconds.";
    if (idle_percent != 0.0) {
      // If the user was calling SleepUntil instead of WaitUntil, this will
      // always be zero; so don't print it in that case.
      KALDI_LOG << "Percentage of time spent idling was " << idle_percent;
    }
    KALDI_LOG << "Longest delay was " << max_delay_ << " seconds for utterance "
              << '\'' << max_delay_utt_ << '\'';
  } else {
    // we have processed each utterance in one chunk.
    // the decoding code will have "pretended to wait" (using WaitUntil())
    // till the end of the utterance before starting to decode it.
    // We want to subtract this waiting time, which is not really
    // of interest in an offline scenario, before printing the
    // real-time factor.
    double real_time_factor = (total_time_taken_ - total_time_waited_) /
        total_audio_;
    KALDI_LOG << "Timing stats: real-time factor for offline decoding was "
              << real_time_factor << " = "
              << (total_time_taken_ - total_time_waited_) << " seconds "
              << " / " << total_audio_ << " seconds.";
  }
}

OnlineTimer::OnlineTimer(const std::string &utterance_id):
    utterance_id_(utterance_id), waited_(0.0), utterance_length_(0.0) { }

void OnlineTimer::WaitUntil(double cur_utterance_length) {
  double elapsed = timer_.Elapsed();
  // it's been cur_utterance_length seconds since we would have
  // started processing this utterance, in a real-time decoding
  // scenario.  We've been actually processing it for "elapsed"
  // seconds, plus we would have been waiting on some kind of
  // semaphore for waited_ seconds.  If we have to wait further
  // at this point, increase "waited_".
  // (I have to think of a better way of explaining this).
  double to_wait = cur_utterance_length - (elapsed + waited_);
  if (to_wait > 0.0)
    waited_ += to_wait;

  utterance_length_ = cur_utterance_length;
}

void OnlineTimer::SleepUntil(double cur_utterance_length) {
  KALDI_ASSERT(waited_ == 0 && "Do not mix SleepUntil with WaitUntil.");
  double elapsed = timer_.Elapsed();

  double to_wait = cur_utterance_length - elapsed;
  if (to_wait > 0.0) {
    Sleep(to_wait);
  }
  utterance_length_ = cur_utterance_length;
}

double OnlineTimer::Elapsed() {
  return timer_.Elapsed() + waited_;
}

void OnlineTimer::OutputStats(OnlineTimingStats *stats) {
  double processing_time = timer_.Elapsed() + waited_,
      wait_time = processing_time - utterance_length_;
  if (wait_time < 0.0) {
    // My first though was to make this a KALDI_ERR, but perhaps
    // clocks can go backwards under some weird circumstance, so
    // let's just make it a warning.
    KALDI_WARN << "Negative wait time " << wait_time
               << " does not make sense.";
  }
  KALDI_VLOG(2) << "Latency " << wait_time << " seconds out of "
                << utterance_length_ << ", for utterance "
                << utterance_id_;
  
  stats->num_utts_++;
  stats->total_audio_ += utterance_length_;
  stats->total_time_taken_ += processing_time;
  stats->total_time_waited_ += waited_;
  if (wait_time > stats->max_delay_) {
    stats->max_delay_ = wait_time;
    stats->max_delay_utt_ = utterance_id_;
  }
}


}  // namespace kaldi

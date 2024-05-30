// online2/online-timing.h

// Copyright 2014   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_TIMING_H_
#define KALDI_ONLINE2_ONLINE_TIMING_H_

#include <string>
#include <vector>
#include <deque>

#include "base/timer.h"
#include "base/kaldi-error.h"

namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{


class OnlineTimer;

/// class OnlineTimingStats stores statistics from timing of online decoding,
/// which will enable the Print() function to print out the average real-time
/// factor and average delay per utterance.  See class OnlineTimer.
class OnlineTimingStats {
 public:
  OnlineTimingStats();
  /// Here, if "online == false" we take into account that the setup was used in
  /// not-really-online mode where the chunk length was the whole file.  We need
  /// to change the way we interpret the stats and print results, in this case.
  void Print(bool online = true);
 protected:
  friend class OnlineTimer;
  int32 num_utts_;
  // all times are in seconds.
  double total_audio_; // total time of audio.
  double total_time_taken_;  // total time spent processing the audio.
  double total_time_waited_; // total time we pretended to wait (but just
                             // increased the waited_ counter)... zero if you
                             // called SleepUntil instead of WaitUntil().
  double max_delay_; // maximum delay at utterance end.
  std::string max_delay_utt_;
};



/// class OnlineTimer is used to test real-time decoding algorithms and evaluate
/// how long the decoding of a particular utterance would take.  The 'obvious'
/// way to evaluate this would be to measure the wall-clock time, and if we're
/// processing the data in chunks, to sleep() until a given chunk would become
/// available in a real-time application-- e.g. say we need to process a chunk
/// that ends half a second into the utterance, we would sleep until half a
/// second had elapsed since the start of the utterance.  In this code we
/// have the option to not actually sleep: we can simulate the effect of
/// sleeping by just incrementing
/// a variable that says how long we would have slept; and we add this to
/// wall-clock times obtained from Timer::Elapsed().
/// The usage of this class will be something like as follows:
/// \code
/// OnlineTimingStats stats;
/// while (.. process different utterances..) {
///   OnlineTimer this_utt_timer(utterance_id);
///   while (...process chunks of this utterance..) {
///      double num_secs_elapsed = 0.01 * num_frames;
///      this_utt_timer.WaitUntil(num_secs_elapsed);
///   }
///   this_utt_timer.OutputStats(&stats);
/// \endcode
/// This assumes that the value provided to the last WaitUntil()
/// call was the length of the utterance.

class OnlineTimer {
 public:
  OnlineTimer(const std::string &utterance_id);

  /// The call to SleepUntil(t) will sleep until cur_utterance_length seconds
  /// after this object was initialized, or return immediately if we've
  /// already passed that time.
  void SleepUntil(double cur_utterance_length);

  /// The call to WaitUntil(t) simulates the effect of sleeping until
  /// cur_utterance_length seconds after this object was initialized; 
  /// but instead of actually sleeping, it increases a counter.
  void WaitUntil(double cur_utterance_length);
  
  /// This call, which should be made after decoding is done,
  /// writes the stats to the object that accumulates them.
  void OutputStats(OnlineTimingStats *stats);

  /// Returns the simulated time elapsed in seconds since the timer was started;
  /// this equals waited_ plus the real time elapsed.
  double Elapsed();
  
 private:
  std::string utterance_id_;
  Timer timer_;
  // all times are in seconds.
  double waited_;
  double utterance_length_;
};


/// @} End of "addtogroup onlinedecoding"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_TIMING_

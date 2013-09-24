// online/online-audio-source.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#ifndef KALDI_ONLINE_ONLINE_AUDIO_SOURCE_H_
#define KALDI_ONLINE_ONLINE_AUDIO_SOURCE_H_

#include <portaudio.h>
#include <pa_ringbuffer.h>

#include "matrix/kaldi-vector.h"

namespace kaldi {

class OnlineAudioSourceItf {
 public:
  // Reads from the audio source, and writes the samples converted to BaseFloat
  // into the vector pointed by "data".
  // The user sets data->Dim() as a way of requesting that many samples.
  // The function returns true if there may be more data, and false if it
  // knows we are at the end of the stream.
  // In case an unexpected and unrecoverable error occurs the function throws
  // an exception of type std::runtime_error (e.g. by using KALDI_ERR macro).
  //
  // NOTE: The older version of this interface had a second paramater - "timeout".
  //       We decided to remove it, because we don't envision usage scenarios,
  //       where "timeout" will need to be changed dynamically from call to call.
  //       If the particular audio source can experience timeouts for some reason
  //       (e.g. the samples are received over a network connection)
  //       we encourage the implementors to configure timeout using a
  //       constructor parameter.
  //       The suggested semantics are: if timeout is used and is greater than 0,
  //       this method has to wait no longer than "timeout" milliseconds before
  //       returning data-- by that time, it will return as much data as it has.
  virtual bool Read(Vector<BaseFloat> *data) = 0;

  virtual ~OnlineAudioSourceItf() { }
};


// OnlineAudioSourceItf implementation using PortAudio to read samples in real-time
// from a sound card/microphone.
class OnlinePaSource : public OnlineAudioSourceItf {
 public:
  typedef int16 SampleType; // hardcoded 16-bit audio
  typedef ring_buffer_size_t rbs_t;

  // PortAudio is initialized here, so it may throw an exception on error
  // "timeout": if > 0, and the acquisition takes more than this number of
  //            milliseconds, Compute() will return the data it has so far
  //            If no data was received until timeout expired, Compute() returns
  //            false (assumes sensible timeout).
  // "sample_rate": the input rate to request from PortAudio
  // "rb_size": requested size of PA's ring buffer - will be round up to
  //           power of 2
  // "report_interval": if not 0, PA ring buffer overflow will be reported
  //                    at every ovfw_msg_interval-th call to Read().
  //                    Putting 0 into this argument disables the reporting.
  OnlinePaSource(const uint32 timeout,
                 const uint32 sample_rate,
                 const uint32 rb_size,
                 const uint32 report_interval);

  // Implementation of the OnlineAudioSourceItf
  bool Read(Vector<BaseFloat> *data);

  // Making friends with the callback so it will be able to access a private
  // member function to delegate the processing
  friend int PaCallback(const void *input, void *output,
                        long unsigned frame_count,
                        const PaStreamCallbackTimeInfo *time_info,
                        PaStreamCallbackFlags status_flags,
                        void *user_data);

  // Returns True if the last call to Read() failed to read the requested
  // number of samples due to timeout.
  bool TimedOut() { return timed_out_; }

  ~OnlinePaSource();

 private:
  // The real PortAudio callback delegates to this one
  int Callback(const void *input, void *output,
               ring_buffer_size_t frame_count,
               const PaStreamCallbackTimeInfo *time_info,
               PaStreamCallbackFlags status_flags);

  uint32 timeout_; // timeout in milliseconds. if > 0, after this many ms. we
                   // give up trying to read data from PortAudio
  bool timed_out_; // True if the last call to Read() failed to obtain the requested
                   // number of samples, because of timeout
  uint32 sample_rate_; // the sampling rate of the input audio
  int32 rb_size_;
  char *ring_buffer_; // points to the actual buffer used by PA to store samples
  PaUtilRingBuffer pa_ringbuf_; // a data structure used to wrap the ring buffer
  PaStream *pa_stream_;
  bool pa_started_; // becomes "true" after "pa_stream_" is started
  uint32 report_interval_; // interval (in Read() calls) to report PA rb overflows
  uint32 nread_calls_; // number of Read() calls so far
  uint32 noverflows_; // number of the ringbuf overflows since the last report
  uint32 samples_lost_; // samples lost, due to PA ring buffer overflow
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlinePaSource);
};

// The actual PortAudio callback - delegates to OnlinePaSource->PaCallback()
int PaCallback(const void *input, void *output,
               long unsigned frame_count,
               const PaStreamCallbackTimeInfo *time_info,
               PaStreamCallbackFlags status_flags,
               void *user_data);


// Simulates audio input, by returning data from a Vector.
// This class is mostly meant to be used for online decoder testing using
// pre-recorded audio
class OnlineVectorSource: public OnlineAudioSourceItf {
 public:
  OnlineVectorSource(const VectorBase<BaseFloat> &input)
      : src_(input), pos_(0) {}

  // Implementation of the OnlineAudioSourceItf
  bool Read(Vector<BaseFloat> *data);

 private:
  Vector<BaseFloat> src_;
  uint32 pos_; // the index of the first element, not yet consumed
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineVectorSource);
};

} // namespace kaldi

#endif // KALDI_ONLINE_ONLINE_AUDIO_SOURCE_H_
